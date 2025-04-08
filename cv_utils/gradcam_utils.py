from pandas.api.types import is_integer_dtype, is_float_dtype
from fastai.vision.all import *
from torch.autograd import Function as TorchFunc
from torch.autograd import Variable


from cv_utils.fastai_utils import prepare_inference_dataloader

def get_plotable_resized_image(X_tensor):
    img = (X_tensor.clamp(0.,1.) * 255.).long() # Fastai's IntToFloatTensor decode
    img = img.permute(1,2,0).cpu().detach().numpy()
    return img

class CustomReLU(TorchFunc):
    """
    Define the custom change to the standard ReLU function necessary to perform guided backpropagation.
    """

    @staticmethod
    def forward(self, x):
        output = torch.addcmul(torch.zeros(x.size()), x, (x > 0).type_as(x))
        self.save_for_backward(x, output)
        return output

    @staticmethod
    def backward(self, dout):
        ##############################################################################
        # Perform a backwards pass as described in  
        # the guided backprop paper (Springenberg et al. 2014).
        # input:
        #   dout is the upstream gradient
        # output:
        #   return downstream gradient
        ##############################################################################

        grad=None
        inp,outp = self.saved_tensors

        backprop_mask = (inp>0).type_as(inp)
        deconv_mask = (dout>0).type_as(dout)

        grad_backprop = torch.addcmul(torch.zeros(inp.size()), dout, backprop_mask)
        grad = torch.addcmul(torch.zeros(inp.size()),grad_backprop,deconv_mask)
        return grad


class GradCam:
    def __init__(self,gc_model,label_names=None):
        """
            model: A pretrained CNN that will be used to compute the gradcam.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gc_model = gc_model.to(self.device)
        self.label_names=label_names
        
    def get_guided_backprop_grad(self, X_tensor, y_tensor):
        """
        Compute a guided backprop visualization using gc_model for images X_tensor and 
        labels y_tensor.

        Input:
        - X_tensor: Input images; Tensor of shape (N, 3, H, W)
        - y: Labels for X; LongTensor of shape (N,)
        - model: A pretrained CNN that will be used to compute the guided backprop.

        Returns:
        - guided backprop: A numpy of shape (N, H, W, 3) giving the guided backprop for 
        the input images.
        """
        self.X_tensor = X_tensor
        self.gc_model.zero_grad()
        for param in self.gc_model.parameters():
            param.requires_grad = True

        if hasattr(self.gc_model,'features'):
            for idx, module in self.gc_model.features._modules.items():
                if module.__class__.__name__ == 'ReLU':
                    self.gc_model.features._modules[idx] = CustomReLU.apply
                elif module.__class__.__name__ == 'Fire':
                    for idx_c, child in self.gc_model.features[int(idx)].named_children():
                        if child.__class__.__name__ == 'ReLU':
                            self.gc_model.features[int(idx)]._modules[idx_c] = CustomReLU.apply


        # Wrap the input tensors in Variables
        X_var = Variable(X_tensor.to(self.device), requires_grad=True)
        y_var = Variable(y_tensor.to(self.device), requires_grad=False)

        outp = self.gc_model(X_var)
        
        # get unnormalized score for correct class
        outp_labels = outp.gather(1,y_var.view(-1,1)) # (N,1)
        outp_labels_sum = torch.sum(outp_labels)

        outp_labels_sum.backward()
        X_grad = X_var.grad # (bs,3,h,w)
        
        guided_bp_grad = X_grad.cpu().detach().numpy().transpose(0,2,3,1)

        #rescale
        self.guided_bp_grad_nonrescale = guided_bp_grad
        gbp_min = guided_bp_grad.min(axis=(1,2,3))[:,None,None,None]
        gbp_max = guided_bp_grad.max(axis=(1,2,3))[:,None,None,None]
        self.guided_bp_grad = (guided_bp_grad - gbp_min) / (gbp_max - gbp_min)
        return self.guided_bp_grad
        
    
    def get_gradcam_value(self, X_tensor, y_tensor, conv_module):
        """
        Input:
        - X_tensor: Input images; Tensor of shape (N, 3, H, W)
        - y: Labels for X; LongTensor of shape (N,)
        - conv_module: a layer in the trained model to calculate GradCAM
        """
        self.X_tensor = X_tensor
        
        self.gc_model.zero_grad()
        for param in self.gc_model.parameters():
            param.requires_grad = True
            
        self.gradient_value = None  # Stores gradient of the module you chose above during a backwards pass.
        self.activation_value = None  # Stores the activation of the module you chose above during a forwards pass.

        def gradient_hook(a, b, gradient):
            self.gradient_value = gradient[0]

        def activation_hook(a, b, activation):
            self.activation_value = activation

        handle1 = conv_module.register_forward_hook(activation_hook)
        handle2 = conv_module.register_backward_hook(gradient_hook)
        
        ##############################################################################
        # Compute a gradcam visualization using gc_model and convolution layer as    #
        # conv_module for images X_tensor and labels y_tensor.                       # 

        
        # Wrap the input tensors in Variables
        X_var = Variable(X_tensor.to(self.device), requires_grad=True)
        y_var = Variable(y_tensor.to(self.device), requires_grad=False)
        
        outp = self.gc_model(X_var)
        # get unnormalized score for correct class
        outp_labels = outp.gather(1,y_var.view(-1,1)) # (N,1)
        outp_labels_sum = torch.sum(outp_labels)
        outp_labels_sum.backward()

        # print(self.activation_value.shape) # (bs,512,13,13)
        grad_value_mean = self.gradient_value.cpu().detach().numpy().mean(axis=(-1,-2),keepdims = True)
        act_value = self.activation_value.cpu().detach().numpy()
        cam = (act_value * grad_value_mean).mean(axis=1) # (bs,13,13)
        cam = np.where(cam >= 0, cam, 0)

        # Rescale GradCam output to fit image.
        cam_scaled = []
        for i in range(cam.shape[0]):
            cam_scaled.append(np.array(Image.fromarray(cam[i]).resize(X_tensor[i, 0, :, :].shape, Image.BICUBIC)))
        cam = np.array(cam_scaled)
        cam -= np.min(cam)
        cam /= np.max(cam)
        
        handle1.remove()
        handle2.remove()

        self.gradcam_val = cam
        return self.gradcam_val


    def plot(self,plot_type,gradcam_val=None,guided_bp_grad=None,X_tensor=None,labels=None,figsize_each=3,fontsize=8,include_original=True,save_to_bytes=False):
        assert plot_type in ['gc','gbp','ggc'], "Plot type must be either:\n\t'gc' for GradCAM\n\t'gbp' for Guided BackProp\n\t'ggc' for Guided GradCAM"
        if X_tensor is None: 
            try: X_tensor = self.X_tensor
            except: raise Exception('Please provide the input tensor X_tensor')
               
        if plot_type in ['gbp','ggc'] and guided_bp_grad is None:
            if not hasattr(self,'guided_bp_grad'): 
                raise Exception('Guided Backprop gradient is not provided/calculated. Run `get_guided_backprop_grad` first')
            else:
                guided_bp_grad = self.guided_bp_grad_nonrescale if plot_type=='ggc' else self.guided_bp_grad
                assert len(X_tensor)==len(guided_bp_grad)
        
        if plot_type in ['gc','ggc'] and gradcam_val is None:
            if not hasattr(self,'gradcam_val'): 
                raise Exception('GradCAM value is not provided/calculated. Run `get_gradcam` first')
            else:
                gradcam_val = self.gradcam_val
                assert len(gradcam_val)==len(X_tensor)
            
        num_cols = 3 if plot_type=='ggc' else 2
        if not include_original: num_cols-=1
        
        fig,axs = plt.subplots(len(X_tensor),num_cols,squeeze=False,figsize=(figsize_each*num_cols,figsize_each*len(X_tensor)))
        # if len(X_tensor)==1: axs = [axs]

        if labels is not None:
            if not isinstance(labels, (list,pd.Series,np.ndarray)):
                labels = [labels for _ in range(len(X_tensor))]
            elif isinstance(labels,(pd.Series,np.ndarray)):
                assert len(labels)==len(X_tensor)
                labels = labels.tolist()
        
        for i in range(len(X_tensor)):
            ax_i=0
            # org img
            img = get_plotable_resized_image(X_tensor[i])
            if include_original:
                axs[i][ax_i].imshow(img)
                axs[i][ax_i].axis('off')
                ax_i+=1
                
            if plot_type=='gbp':
                axs[i][ax_i].imshow(guided_bp_grad[i])
            else:
                # gradcam
                gc = gradcam_val[i]
                img_gc = img + (matplotlib.cm.jet(gc)[:,:,:3]*255)
                img_gc = img_gc / np.max(img_gc)
                axs[i][ax_i].imshow(img_gc)
                   
            if labels is not None:
                axs[i][ax_i].set_title(f'{labels[i]}', fontsize=fontsize)
            axs[i][ax_i].axis('off')
            ax_i+=1

            if plot_type=='ggc':
                # guided gc
                bp = guided_bp_grad[i]
                img_both = np.expand_dims(gradcam_val[i], axis=2)*bp
                
                #rescale
                img_both = (img_both - img_both.min()) / (img_both.max() - img_both.min())
                
                axs[i][ax_i].imshow(img_both)
                axs[i][ax_i].axis('off')
            
        # Display the plot
        plt.tight_layout()
        if save_to_bytes:                        
            with io.BytesIO() as buffer:  # use buffer memory
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                img = buffer.getvalue()
                plt.close()
                return img
        else:
            plt.show()

    def get_gradcam_images(self,
                            detections,
                            input_container_sas=None,
                            item_tfms=Resize(750),
                            batch_size=1,
                            plot_type='gc',
                            path_column='file',
                            y_column='pred_1',
                            bbox_column=None, # 'detection_bbox'
                            prob_column=None, # 'prob_1'
                            conv_module=None, # _conv_head
                            figsize_each=6,
                            fontsize=12,
                            save_to_bytes=False
                           ):
        if len(detections)==0:
            return []
        assert isinstance(detections,pd.DataFrame), '`detections` must be a DataFrame'
        assert path_column in detections.columns, f'`detections` dataframe must contain a column `{path_column}` containing absolute img file path'
        assert y_column in detections.columns and is_integer_dtype(detections[y_column]),\
            f"`detections` dataframe must contain a column `{y_column}` containing INTEGER values of the label to calculate GradCAM/Guided Backprop from"
        if prob_column is not None:
            assert prob_column in detections.columns and is_float_dtype(detections[prob_column]),\
            f"`detections` dataframe must contain a column `{prob_column}` containing FLOAT prediction probabilities"
            
        if bbox_column in detections.columns:
            test_inps = list(zip(detections[path_column], detections[bbox_column]))
        else:
            test_inps = detections[path_column].tolist()

        detection_labels = detections[y_column]
        if self.label_names is not None:
            detection_labels = detections[y_column].map({i:v for i,v in enumerate(self.label_names)})

            
        test_dl,_ = prepare_inference_dataloader(test_inps,
                                                 input_container_sas=input_container_sas,
                                                 item_tfms=item_tfms,
                                                 batch_size=batch_size)
        
        test_iter = iter(test_dl.valid)
        img_bytes=[]
        for i,X_tensor in enumerate(test_iter):
            _s = i*batch_size
            _e = (i+1)*batch_size
            X_tensor = X_tensor[0]
            y_tensor = torch.LongTensor(detections[y_column].values[_s:_e])

            y_label = detection_labels[_s:_e]
            if prob_column is not None:
                y_prob = detections[prob_column].values[_s:_e]
                y_label = [f"{a}: {b:.3f}" for a,b in zip(y_label,y_prob)]
            

            if plot_type in ['gbp','ggc']:
                _ = self.get_guided_backprop_grad(X_tensor,y_tensor)
            if plot_type in ['gc','ggc']:
                _ = self.get_gradcam_value(X_tensor,y_tensor,conv_module=conv_module)
            img_byte = self.plot(plot_type=plot_type,
                               labels=y_label,
                               figsize_each=figsize_each,
                               fontsize=fontsize,
                               save_to_bytes=save_to_bytes
                              )
            if img_byte is not None: 
                img_bytes.append(img_byte)

        return img_bytes
    


def get_gradcam_results(scorer, # DetectAndClassify object
                        file_paths, # list of absolute file paths to images
                        input_container_sas=None,
                        plot_type='gc',
                        figsize_each=5, # plt fig size for each image
                       ):
    if isinstance(file_paths,str):
        file_paths = [file_paths]
    detections = scorer.predict(file_paths,
                                input_container_sas=input_container_sas,
                                convert_to_json=False,
                                pred_topn=1,
                                n_workers=1
                               )
    detections = detections[detections['failure'].isna() & (detections['detection_category'].astype(int).isin([1]))].copy().reset_index(drop=True)
    gc = GradCam(scorer.class_inference.model,label_names=scorer.class_inference.label_info)
    
    results = gc.get_gradcam_images(detections,
                                    input_container_sas=input_container_sas,
                                    item_tfms=scorer.class_inference.item_tfms,
                                    batch_size=1,
                                    plot_type=plot_type,
                                    path_column='file',
                                    y_column='pred_1',
                                    bbox_column='detection_bbox',
                                    prob_column='prob_1',
                                    conv_module=scorer.class_inference.model._conv_head,
                                    save_to_bytes=True,
                                    figsize_each=figsize_each
                                   )
    return results,detections
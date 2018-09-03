# Small-increase-in-DeepFake
Small increase in face generation replacement
## Small increase
&#8195;'This is a small increase in face replacement (deepfakes): In the original program, keypoint detection (mtcnn) is used when the faces are aligned, then the image is rotated and the interpolation algorithm is used. This is reasonable but may cause additional errors (due to the interpolation algorithm), so the face alignment is divided into two phases, first key point detection, using mtcnn, then rotating the entire picture (face and horizontal lines of the face) Angle), and finally face re-extraction of the rotated picture, which can completely avoid the error of face alignment.<br>
## Running process
1.&#8195;Execute mtcnn to find the offset angle<br>2.&#8195;Rotate the picture to reposition the face<br>3.&#8195;Training generation network<br>4.&#8195;Rotate the original image at the same angle to replace the face.<br>5.&#8195;Reverse the replaced image and extract the final replacement map.<br>
## Instructions for use
1.&#8195;Configure and run first_angle.py second_align.py third_input_image.py in order to get the input image pair.<br>2.&#8195;Run GAN_model.py to start training the model. The optional operations are 'training' and 'testing'.<br>3.&#8195;The relevant configuration options are path and learning rate, loss weight, batch size, etc. They are very common and do not have high difficulty.<br>

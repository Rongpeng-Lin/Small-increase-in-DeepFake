import click
from struct import *

class GAN_model:
    def __init__(self,batch, epoch, DA_lr, DB_lr, GA_lr, GB_lr, l1_weight, gan_weight, save_iters, output_dir, faceA_dir, faceB_dir):
        self.batch = batch
        self.epoch = epoch
        self.A_wrap = tf.placeholder(tf.float32,[batch,64,64,3],'A_wrap')
        self.A_raw = tf.placeholder(tf.float32,[batch,64,64,3],'A_raw')
        self.B_wrap = tf.placeholder(tf.float32,[batch,64,64,3],'B_wrap')
        self.B_raw = tf.placeholder(tf.float32,[batch,64,64,3],'B_raw')
        
        self.DA_lr = DA_lr
        self.DB_lr = DB_lr
        self.GA_lr = GA_lr
        self.GB_lr = GB_lr
        self.l1_weight = l1_weight
        self.gan_weight = gan_weight
        self.save_iters = save_iters
        self.output_dir = output_dir
        
        self.faceA_dirs = [faceA_dir+name for name in os.listdir(faceA_dir)] if faceA_dir else None
        self.faceB_dirs = [faceB_dir+name for name in os.listdir(faceB_dir)] if faceB_dir else None
        self.forward()
        
    def G_net(self,fA,fB): # fA: 64*64*3  fB: 64*64*3
        encode_A = Encoder('encode',fA,False)
        print('Encoder: ',[i.value for i in encode_A.get_shape()])
        encode_B = Encoder('encode',fB,True)
        decode_A = Decoder_ps('decode_A',encode_A,False)
        print('Decoder: ',[i.value for i in decode_A.get_shape()])
        decode_B = Decoder_ps('decode_B',encode_B,False)
        return decode_A,decode_B
        
    def DA_net(self,image,re_use):
        with tf.variable_scope('DA_net',reuse=re_use):
            block1 = conv_block_d('block1',image,64)
            block2 = conv_block_d('block2',block1,128)
            block3 = conv_block_d('block3',block2,256)
            out = conv('out',block3,4*4,1,1,False,True)
            return out
    
    def DB_net(self,image,re_use):
        with tf.variable_scope('DB_net',reuse=re_use):
            block1 = conv_block_d('block1',image,64)
            block2 = conv_block_d('block2',block1,128)
            block3 = conv_block_d('block3',block2,256)
            out = conv('out',block3,4*4,1,1,False,True)
            return out
        
    def forward(self):
        self.deA,self.deB = self.G_net(self.A_wrap,self.B_wrap)
        self.realA_out = self.DA_net(self.A_raw,False)
        self.deA_out = self.DA_net(self.deA,True)
        self.realB_out = self.DB_net(self.B_raw,False)
        self.deB_out = self.DB_net(self.deB,True)
        
    def read_image(self,num,Train,Test):
        zeros1 = np.zeros([self.batch,64,128,3])
        zeros2 = np.zeros([self.batch,64,128,3])
        if Train:
            Adirs = self.faceA_dirs[int(num*self.batch):(int(num*self.batch)+self.batch)]
            Bdirs = self.faceB_dirs[int(num*self.batch):(int(num*self.batch)+self.batch)]
            for i in range(self.batch):
                ima = cv.imread(Adirs[i])
                imb = cv.imread(Bdirs[i])
                ima,imb = ima.astype(np.float32),imb.astype(np.float32)
                zeros1[i,:,:,:] = ima
                zeros2[i,:,:,:] = imb
            zeros1 = zeros1/127.5 - 1
            zeros2 = zeros2/127.5 - 1
            return Adirs,Bdirs,zeros1[:,:,0:64,:],zeros1[:,:,64:128,:],zeros2[:,:,0:64,:],zeros2[:,:,64:128,:]
        if Test=='fA':
            ima = cv.imread(self.faceA_dirs[0])
            zeros1[0,:,:,:] = ima
            zeros1 = zeros1/127.5 - 1
            return self.faceA_dirs[0],zeros1[:,:,0:64,:],zeros1[:,:,64:128,:],zeros2[:,:,0:64,:],zeros2[:,:,64:128,:]
        if Test=='fB':
            ima = cv.imread(self.faceB_dirs[0])
            zeros2[0,:,:,:] = ima
            zeros2 = zeros2/127.5 - 1
            return self.faceB_dirs[0],zeros2[:,:,0:64,:],zeros2[:,:,64:128,:],zeros1[:,:,0:64,:],zeros1[:,:,64:128,:]
            
    def file_and_name(self,L):
        file = L.split('/')[-2]
        name = L.split('/')[-1]
        return file,name
    
    def color_255(self,image):
        image255 = (image+1)*127.5
        return image255
        
    def save_image(self,epo,A_dirs,B_dirs,decode_A,decode_B):
        for i in range(self.batch):
            a_dir,b_dir = A_dirs[i],B_dirs[i]
            a_file,a_name = self.file_and_name(a_dir)
            b_file,b_name = self.file_and_name(b_dir)
            if not os.path.exists(self.output_dir+'/'+a_file):
                os.makedirs(self.output_dir+'/'+a_file)
            if not os.path.exists(self.output_dir+'/'+b_file):
                os.makedirs(self.output_dir+'/'+b_file)
            a_new_name = a_name.split('.')[0]+'_'+str(epo)+'.jpg'
            b_new_name = a_name.split('.')[0]+'_'+str(epo)+'.jpg'
            a_255 = self.color_255(decode_A[i,:,:,:])
            b_255 = self.color_255(decode_B[i,:,:,:])
            cv.imwrite(self.output_dir+'/'+a_file+'/'+a_new_name,a_255)
            cv.imwrite(self.output_dir+'/'+b_file+'/'+b_new_name,b_255)
 
    def train(self):
        self.l1_loss_GA = tf.reduce_mean(self.deA-self.A_raw)
        self.l1_loss_GB = tf.reduce_mean(self.deB-self.B_raw)
        self.DA_loss = tf.reduce_mean(0.5*tf.squared_difference(self.realA_out,1)) + tf.reduce_mean(0.5*tf.square(self.deA_out))                
        self.GAN_GA_loss = tf.reduce_mean(0.5*tf.squared_difference(self.deA,1))
        self.DB_loss = tf.reduce_mean(0.5*tf.squared_difference(self.realB_out,1)) + tf.reduce_mean(0.5*tf.square(self.deB_out))
        self.GAN_GB_loss = tf.reduce_mean(0.5*tf.squared_difference(self.deB,1))
        self.GA_loss = self.l1_weight * self.l1_loss_GA + self.gan_weight * self.GAN_GA_loss
        self.GB_loss = self.l1_weight * self.l1_loss_GB + self.gan_weight * self.GAN_GB_loss
        
        DA_vars = [var for var in tf.all_variables() if 'DA_net' in var.name]
        DB_vars = [var for var in tf.all_variables() if 'DB_net' in var.name]
        GA_vars = [var for var in tf.all_variables() if 'encode' or 'decode_A' in var.name]
        GB_vars = [var for var in tf.all_variables() if 'encode' or 'decode_B' in var.name]
        
        DA_optim = tf.train.AdamOptimizer(self.DA_lr).minimize(loss=self.DA_loss,var_list=DA_vars)
        DB_optim = tf.train.AdamOptimizer(self.DB_lr).minimize(loss=self.DB_loss,var_list=DB_vars)
        GA_optim = tf.train.AdamOptimizer(self.GA_lr).minimize(loss=self.GA_loss,var_list=GA_vars)
        GB_optim = tf.train.AdamOptimizer(self.GB_lr).minimize(loss=self.GB_loss,var_list=GB_vars)       
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            graph = tf.summary.FileWriter('E:/model/', graph=sess.graph)
            Saver = tf.train.Saver(max_to_keep=10)
            for i in range(self.epoch):
                for j in range(min(len(self.faceA_dirs),len(self.faceB_dirs))//self.batch):
                    a_dirs,b_dirs,wrap_a,raw_a,wrap_b,raw_b = self.read_image(j,True,None)
                    Dict = {self.A_raw:raw_a,self.A_wrap:wrap_a,self.B_raw:raw_b,self.B_wrap:wrap_b}         
                    _ = sess.run([DA_optim],feed_dict=Dict)
                    _ = sess.run([DB_optim],feed_dict=Dict)
                    _,decodeA = sess.run([GA_optim,self.deA],feed_dict=Dict)
                    _,decodeB = sess.run([GB_optim,self.deB],feed_dict=Dict)         
                    if (i%self.save_iters)==0:
                        self.save_image(i,a_dirs,b_dirs,decodeA,decodeB)
                        print('save images at epoch %d'%(i))
                Saver.save(sess,save_path='E:/model/deepfake.ckpt',global_step=i)
                print('成了：',i)
        
    def test(self,Test,txt_dir,ckpt_path,output_dir):
        _, image_raw, image_wrap, zero_raw, zero_wrap = self.read_image(0,False,Test)
        raw_image_dir,angle,box = angle_and_box(txt_dir)
        
        if Test=='fA':
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                GA_vars = [var for var in tf.all_variables() if 'encode' or 'decode_A' in var.name]
                GB_vars = [var for var in tf.all_variables() if 'encode' or 'decode_B' in var.name]
                G_vars = GA_vars + GB_vars
                Saver = tf.train.Saver(var_list=G_vars)
                Saver.restore(sess,ckpt_path)
                print('load success')
                Dict = {self.A_raw:zero_raw,self.A_wrap:zero_wrap,self.B_raw:image_raw,self.B_wrap:image_wrap}
                fake = sess.run(self.deB,feed_dict=Dict)
                print('get_fake')
                final_image = get_final_image(raw_image_dir,angle,box,fake[0,:,:,:])
                cv.imwrite(output_dir+'/'+'fake_face.jpg',final_image)
                print('write done')
                return True
        else:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                GA_vars = [var for var in tf.all_variables() if 'encode' or 'decode_A' in var.name]
                GB_vars = [var for var in tf.all_variables() if 'encode' or 'decode_B' in var.name]
                G_vars = GA_vars + GB_vars
                Saver = tf.train.Saver(var_list=G_vars)
                Saver.restore(sess,ckpt_path)
                Dict = {self.A_raw:image_raw,self.A_wrap:image_wrap,self.B_raw:zero_raw,self.B_wrap:zero_wrap}
                fake = sess.run(self.deA,feed_dict=Dict)


@click.command()
@click.option('--batch',
              type=click.INT,
              default=16,
              help='Batch_size.')
@click.option('--epoch',
              type=click.INT,
              default=100,
              help='Epoch.')
@click.option('--DA_lr',
              type=click.FLOAT,
              default=0.0001,
              help='Learning rate of DA.')
@click.option('--DB_lr',
              type=click.FLOAT,
              default=0.0001,
              help='Learning rate of DB.')
@click.option('--GA_lr',
              type=click.FLOAT,
              default=0.0001,
              help='Learning rate of GA.')
@click.option('--GB_lr',
              type=click.FLOAT,
              default=0.0001,
              help='Learning rate of GB.')
@click.option('--l1_weight',
              type=click.FLOAT,
              default=5.0,
              help='Learning rate of GB.')
@click.option('--gan_weight',
              type=click.FLOAT,
              default=5.0,
              help='L1 loss weight.')
@click.option('--save_iters',
              type=click.INT,
              default=5,
              help='The period in which the image is saved.')
@click.option('--output_dir',
              type=click.STRING,
              default='./data/pictures',
              help='Output image path.')
@click.option('--faceA_dir',
              type=click.STRING,
              default='./data/FaceA/',
              help='Face A images path.')
@click.option('--faceB_dir',
              type=click.STRING,
              default='./data/FaceB/',
              help='Face B images path.')
@click.option('--faceB_dir',
              type=click.STRING,
              default='./data/FaceB/',
              help='Face B images path.')
@click.option('--To_train',
              type=click.BOOL,
              default=True,
              help='If train.')
@click.option('--Test',
              type=click.STRING,
              default='fA',
              help='Which net to test.')
@click.option('--txt_dir',
              type=click.STRING,
              default='./data/newbox.txt
              help='Image information.')
@click.option('--ckpt_path',
              type=click.STRING,
              default='./data/model.ckpt-100',
              help='Ckpt path.')
@click.option('--fake_dir',
              type=click.STRING,
              default='./data/fakeface/',
              help='Final fake face.')

def main(batch, epoch, DA_lr, DB_lr, GA_lr, GB_lr, l1_weight, gan_weight, save_iters, output_dir, faceA_dir, faceB_dir, To_train, Test, txt_dir, ckpt_path, fake_dir):   
    gan = GAN_model(batch, epoch, DA_lr, DB_lr, GA_lr, GB_lr, l1_weight, gan_weight, save_iters, output_dir, faceA_dir, faceB_dir)
    if To_train: 
        gan.train()
    else:
        gan.test(Test,txt_dir,ckpt_path,fake_dir)
        
if __name__ == '__main__':
    main()   

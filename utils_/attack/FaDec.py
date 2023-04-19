import numpy as np



class FaDec:
    def __init__(self, data, model, 
                 jumps=([0.1,0.2,0.4,0.8,1.0,2,.0,4.0,8.0])):
        
        self.count = 0
        
        self.data = data
        self.model = model
        
        self.x_min = 0
        self.x_max = 1
        
        self.jumps = jumps
        
        return
    
    
    def query(self, x_in):
        self.count += 1
        x_in = np.expand_dims(x_in, axis=0)
        return self.model.decision(x_in)
    
    
    def fadec_attack(self, f_image, r_image, error=0.1, out_prob=0.2):
        
        m_image = (f_image+r_image)/2
        
        adv_image = self.get_boundary_image(r_image, m_image, f_image, error=error, out_prob=out_prob)
        
        for k in range(10):
            noise = np.random.randint(self.x_min, self.x_max, (r_image.shape[0], r_image.shape[1], 3))
            adv_image_del = np.clip(adv_image + noise, self.x_min, self.x_max)
            adv_image_del = self.get_boundary_image(r_image, adv_image_del, f_image, error=error, out_prob=out_prob)
            
            d1 = self.compute_diff(adv_image, f_image)
            d2 = self.compute_diff(adv_image_del, f_image)
            print("************************")
            print("The iteration number: ", k)
            sign = np.sign(d2-d1)
            diff = sign*adv_image_del-sign*adv_image
            
            for j in self.jumps:
                adv_image2 = self.get_boundary_image(r_image, adv_image+j*diff, f_image, error=error, out_prob=out_prob)
                if self.compute_diff(adv_image2,f_image) < self.compute_diff(adv_image,f_image):
                    adv_image3 = np.copy(adv_image2)
            
            adv_image = np.copy(adv_image3)
            print("Distance change = ", self.compute_diff(adv_image, f_image) - d1,
                  " :: The output decision is = ", self.query(adv_image))
        
        return adv_image


    def compute_diff(self, img1, img2):
        return np.mean(np.abs(img1-img2))


    def get_boundary_image(self, 
                           real_image, mid_image, fake_image, 
                           error=10, out_prob=0.5, 
                           verbose=False, 
                           type_image='real'):
        
        out_d = self.query(mid_image)
        
        if out_d > out_prob:
            fake_image = np.copy(mid_image)
        else:
            real_image = np.copy(mid_image)
        
        diff = self.compute_diff(real_image,fake_image)
        mid_image = (fake_image+real_image)/2

        while diff>error:
            out_d = self.query(mid_image)
            
            if out_d > out_prob:
                fake_image = np.copy(mid_image)
            else:
                real_image = np.copy(mid_image)
            
            if diff > self.compute_diff(real_image,fake_image):
                diff = self.compute_diff(real_image,fake_image)
            else:
                print("######## There seems to be some problem ########")
            
            mid_image = (fake_image+real_image)/2
            
            if verbose==True:
                print("***********")
                print(" New diff = ",diff,". The detection prob. is ", out_d, ".")
        
        if type_image=='real':
            return real_image
        else:
            return fake_image
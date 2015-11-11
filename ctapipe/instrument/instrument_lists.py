def clear_lists_camera():
    
    del pixel_posX[:]
    del pixel_posY[:]
    del pixel_posZ[:]
    del pixel_id[:]
    del camera_class[:]
    del camera_fov[:]
    channel_num = 0
     
def clear_lists_optics():
    
    del mirror_area[:]
    del mirror_number[:]
    del focal_length[:]

def clear_lists_telescope():
    
    del telescope_id[:]
    telescope_num = 0
    del telescope_posX[:]
    del telescope_posY[:]
    del telescope_posZ[:]

#Telescope
telescope_id = []
telescope_num = 0
telescope_posX = []
telescope_posY = []
telescope_posZ = []

#Camera
pixel_posX = []
pixel_posY = []
pixel_posZ = []
pixel_id = []
camera_class = []
camera_fov = []

#Optics
mirror_area = []
mirror_number = []
focal_length = []

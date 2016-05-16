import skimage.io
import numpy as np
import glob
import random
import os
import shutil
import warnings

# size of the cropped images
CROP_WIDTH = 121
CROP_HEIGHT = 121

def generate_crops(img_files, crops_dir, txt_dir):
  # make dirs if it doesn't exist
  if not os.path.exists(crops_dir):
    os.makedirs(crops_dir)

  # initialize the cropped image
  img_crop = np.zeros((CROP_HEIGHT,CROP_WIDTH,3)).astype(np.uint8)

  # open file to write the positive samples names and labels
  f = open(txt_dir,'w')
  for i in range(len(img_files)):
    print "Processing the %d/%d image" %(i+1,len(img_files))
    img_file_name = img_files[i]
    gt_file_name = img_file_name[:-4] + '.txt'
    img = skimage.io.imread(img_file_name)
    gts = np.loadtxt(gt_file_name, delimiter='\t')
    IMG_HEIGHT = img.shape[0]
    IMG_WIDTH = img.shape[1]

    # padded img, since some of the sliding window lies partially outside the image boundary
    padded_img = np.zeros((IMG_HEIGHT+CROP_HEIGHT-1, IMG_WIDTH+CROP_WIDTH-1,3)).astype(np.uint8)
    padded_img[CROP_HEIGHT/2:IMG_HEIGHT+CROP_HEIGHT/2,CROP_WIDTH/2:IMG_WIDTH+CROP_WIDTH/2,:] = img    # pad the middle
    padded_img[CROP_HEIGHT/2:IMG_HEIGHT+CROP_HEIGHT/2,0:CROP_WIDTH/2,:] = img[:,CROP_WIDTH/2:0:-1,:]  # pad the left    
    padded_img[CROP_HEIGHT/2:IMG_HEIGHT+CROP_HEIGHT/2,IMG_WIDTH+CROP_WIDTH/2:IMG_WIDTH+CROP_WIDTH,:] = img[:,IMG_WIDTH-2:IMG_WIDTH-CROP_WIDTH/2-2:-1,:]    # pad the right
    padded_img[:CROP_HEIGHT/2,:,:] = padded_img[CROP_HEIGHT-1:CROP_HEIGHT/2:-1,:,:]    # pad the top
    padded_img[IMG_HEIGHT+CROP_HEIGHT/2:IMG_HEIGHT+CROP_HEIGHT,:,:] = padded_img[IMG_HEIGHT+CROP_HEIGHT/2-2:IMG_HEIGHT-2:-1,:,:]    # pad the bottom
   
    # get the positive sliding windows and save them 
    for j in range(gts.shape[0]):
      gt_y = gts[j,1]
      gt_x = gts[j,0]
      start_y = gt_y - 1
      start_x = gt_x - 1

      img_crop = padded_img[start_y:start_y+CROP_HEIGHT, start_x:start_x+CROP_WIDTH,:]
      crop_name = crops_dir+img_file_name.split('/')[-1][:-4]+'_crop_pos%d.tif'%j
      skimage.io.imsave(crop_name, img_crop)
      f.write(crop_name + ' 1\n')
  f.close()

def generate_random_negative_crops(img_files, crops_dir, txt_dir):
  # make dirs if necessary
  if not os.path.exists(crops_dir):
    os.makedirs(crops_dir)
  
  # initialize the cropped image
  img_crop = np.zeros((CROP_HEIGHT,CROP_WIDTH,3)).astype(np.uint8)
  f = open(txt_dir,'w')
  for i in range(len(img_files)):
    print "Processing the %d/%d image" %(i+1,len(img_files))
    img_file_name = img_files[i]
    gt_file_name = img_file_name[:-4] + '.txt'
    img = skimage.io.imread(img_file_name)
    gts = np.loadtxt(gt_file_name, delimiter='\t')
    IMG_HEIGHT = img.shape[0]
    IMG_WIDTH = img.shape[1]
    
    padded_img = np.zeros((IMG_HEIGHT+CROP_HEIGHT-1, IMG_WIDTH+CROP_WIDTH-1,3)).astype(np.uint8)
    mask = np.ones((IMG_HEIGHT+CROP_HEIGHT-1, IMG_WIDTH+CROP_WIDTH-1)).astype(np.uint8)
    padded_img[CROP_HEIGHT/2:IMG_HEIGHT+CROP_HEIGHT/2,CROP_WIDTH/2:IMG_WIDTH+CROP_WIDTH/2,:] = img    # pad the middle
    padded_img[CROP_HEIGHT/2:IMG_HEIGHT+CROP_HEIGHT/2,0:CROP_WIDTH/2,:] = img[:,CROP_WIDTH/2:0:-1,:]  # pad the left    
    padded_img[CROP_HEIGHT/2:IMG_HEIGHT+CROP_HEIGHT/2,IMG_WIDTH+CROP_WIDTH/2:IMG_WIDTH+CROP_WIDTH,:] = img[:,IMG_WIDTH-2:IMG_WIDTH-CROP_WIDTH/2-2:-1,:]    # pad the right
    padded_img[:CROP_HEIGHT/2,:,:] = padded_img[CROP_HEIGHT-1:CROP_HEIGHT/2:-1,:,:]    # pad the top
    padded_img[IMG_HEIGHT+CROP_HEIGHT/2:IMG_HEIGHT+CROP_HEIGHT,:,:] = padded_img[IMG_HEIGHT+CROP_HEIGHT/2-2:IMG_HEIGHT-2:-1,:,:]    # pad the bottom
    
    for j in range(gts.shape[0]):
      gt_y = gts[j,1]
      gt_x = gts[j,0]
      start_y = gt_y + CROP_HEIGHT/2 - 11
      start_x = gt_x + CROP_WIDTH/2 - 11
      mask[start_y:start_y+21, start_x:start_x+21] = 0
      #padded_img[start_y:start_y+21, start_x:start_x+21,:] = 0

      #crop_name = crops_dir+img_file_name.split('/')[-1][:-4]+'_crop%d.tif'%i
      #skimage.io.imsave(crop_name, img_crop)
      #f.write(crop_name + ' 0\n')      
    for j in range(gts.shape[0]):
      selected_y = np.random.randint(0,IMG_HEIGHT) + CROP_HEIGHT/2
      selected_x = np.random.randint(0,IMG_WIDTH) + CROP_HEIGHT/2
      overlap = False
      
      # check if the random negative window center overlap with the positve window cencer, if overlap, then generate another one randomly
      for y in range(selected_y-10,selected_y+11):
        for x in range(selected_x-10,selected_x+11):
          if mask[y,x] == 0:
            overlap = True
      mask[selected_y,selected_x] == 0
      while overlap:
        selected_y = np.random.randint(0,IMG_HEIGHT) + CROP_HEIGHT/2
        selected_x = np.random.randint(0,IMG_WIDTH) + CROP_HEIGHT/2
        overlap = False
        for y in range(selected_y-10,selected_y+11):
          for x in range(selected_x-10,selected_x+11):
            if mask[y,x] == 0:
             # print mask[y,x]
              overlap = True
      
      # if the sliding window is chosen, we won't choose it again
      mask[selected_y-10:selected_y+11, selected_x-10:selected_x+11] = 0
      img_crop = padded_img[selected_y-CROP_HEIGHT/2:selected_y+CROP_HEIGHT/2+1, selected_x-CROP_WIDTH/2:selected_x+CROP_WIDTH/2+1]

      crop_name = crops_dir+img_file_name.split('/')[-1][:-4]+'_crop_neg%d.tif'%j
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skimage.io.imsave(crop_name, img_crop)
      f.write(crop_name + ' 0\n')
    #skimage.io.imsave('/home/bkong/%d.tif' %i, padded_img)
  f.close()

# merge positive and negative label files
def merge_txt(file1, file2, merged_file):
  f = open(file1,'r')
  lines = f.readlines()
  f.close()
  f = open(file2,'r')
  lines.extend(f.readlines())
  f.close()
  os.remove(file1)
  os.remove(file2)
  f = open(merged_file,'w')
  np.random.shuffle(lines)
  for line in lines:
    f.write(line)
  f.close()

# generate training, testing & validation patients
def generate_patients_set(img_dir,crop_dir):
  C1_imgs = glob.glob(img_dir+'C1*.tif')
  C2_imgs = glob.glob(img_dir+'C2*.tif')

  patient_C1 = {}
  patient_C2 = {}

  for img in C1_imgs:
    patient_C1[img.split('/')[-1].split('_ROI')[0]] = 0
  for img in C2_imgs:
    patient_C2[img.split('/')[-1].split('_ROI')[0]] = 0

  patient_C1 = patient_C1.keys()
  patient_C2 = patient_C2.keys()
  
  # the first 10 patients for C1 and C2 is training
  #random.shuffle(patient_C1)
  #random.shuffle(patient_C2)
  train_patients = patient_C1[0:10]
  train_patients.extend(patient_C2[0:10])
  val_patients = patient_C1[10:15]
  val_patients.extend(patient_C2[10:15])
  test_patients = patient_C1[15:20]
  test_patients.extend(patient_C2[15:20])
  
  # move the image and ground truth files to the corresponding folders
  if not os.path.exists(crop_dir+'train_patient'):
    os.makedirs(crop_dir+'train_patient')
  if not os.path.exists(crop_dir+'val_patient'):
    os.makedirs(crop_dir+'val_patient')
  if not os.path.exists(crop_dir+'test_patient'):
    os.makedirs(crop_dir+'test_patient')
  for patient in train_patients:
    files = glob.glob(img_dir+patient+'*')
    for filename in files:
      shutil.copy(filename,crop_dir+'train_patient')
  for patient in val_patients:
    files = glob.glob(img_dir+patient+'*')
    for filename in files:
      shutil.copy(filename,crop_dir+'val_patient')
  for patient in test_patients:
    files = glob.glob(img_dir+patient+'*')
    for filename in files:
      shutil.copy(filename,crop_dir+'test_patient')
  return train_patients,val_patients,test_patients

if __name__ == '__main__':
  GT_DIR = '/home/bink/Data/Data_GroundTruth/'
  CROP_DIR = '/home/bink/Data/crops/'
  C1_imgs = glob.glob(GT_DIR + 'C1*.tif')
  C2_imgs = glob.glob(GT_DIR + 'C2*.tif')
  train_patients, val_patients, test_patients = generate_patients_set(GT_DIR,CROP_DIR)
  
  train_imgs = []
  print "Generating positive examples for training..."
  for patient in train_patients:
    train_imgs.extend(glob.glob(GT_DIR+patient+'*.tif'))
  generate_crops(train_imgs, CROP_DIR+'train/pos/', CROP_DIR+'train1.txt')
  print "Generating negative examples for training..."
  generate_random_negative_crops(train_imgs, CROP_DIR+'train/neg/', CROP_DIR+'train2.txt')
  merge_txt(CROP_DIR+'train1.txt', CROP_DIR+'train2.txt', CROP_DIR+'train.txt')

  print "Generating positive examples for validation..."
  val_imgs = []
  for patient in val_patients:
    val_imgs.extend(glob.glob(GT_DIR+patient+'*.tif'))
  generate_crops(val_imgs, CROP_DIR+'val/pos/', CROP_DIR+'val1.txt')
  print "Generating negative examples for validation..."
  generate_random_negative_crops(val_imgs, CROP_DIR+'val/neg/', CROP_DIR+'val2.txt')
  merge_txt(CROP_DIR+'val1.txt',CROP_DIR+'val2.txt',CROP_DIR+'val.txt')
  #print "Generating positive examples for testing..."
  #test_imgs = []
  #for patient in test_patients:
  #  test_imgs.extend(glob.glob(GT_DIR+patient+'*.tif'))
  #generate_crops(test_imgs, CROP_DIR+'test/', CROP_DIR+'test.txt')
  

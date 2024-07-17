import cv2
import numpy as np

def show_image_series(window_name:str,images:list,txt:list) -> None:
    frameWidth, frameHeight = 1200, 1200
    rows,cols = 3,2
    Width, Height = int(frameWidth / cols),int(frameHeight / rows)
    image_to_show =  np.zeros((frameWidth, frameHeight,3),dtype=np.uint8)
    print(image_to_show.shape)
    H_count=0
    W_counter= 0
    for num,(img,text) in enumerate(zip(images,txt)):
        img = cv2.resize(img, (Height,Width))
        print('num 1 Weigth: {},{}'.format(W_counter * Width, (W_counter + 1) * Width))
        print('num 1 Height: {},{}\n'.format(H_count*Height,(H_count+1)*Height))
        if len(img.shape)<3 :
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 2, cv2.LINE_AA)
        image_to_show [W_counter*Width:(W_counter+1)*Width,H_count*Height:(H_count+1)*Height,:] = img
        H_count += 1
        if (num+1) % 3 ==0 :
            W_counter += 1
            H_count = 0

    cv2.imshow(window_name,image_to_show)

def preProcessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),0)
    imgCanny = cv2.Canny(imgBlur ,75,200)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)
    return imgThres

def getContours(img):
    biggest = np.array([])
    Area_max = 0
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    biggest_counter = np.array([[0,0],[0,0]],int)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>1000:
            ''' Find shape of counter based on implementation of Douglas-Peucker algorithm'''
            epsilon = 0.1 * cv2.arcLength(cnt, True) # the length that we can deny
            side_num = cv2.approxPolyDP(cnt,epsilon,True)
            if area > Area_max and len(side_num) == 4:
                Area_max = area
                biggest_counter = cnt

    return biggest_counter

def get_corners (myPoints):
    # define new set of points
    myPointsNew = np.zeros((4,1,2),np.int32)
    # find 4 corner points among all points
    add = myPoints.sum(axis=1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1]= myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew

def Transform(img,biggest):
    #get the size of image
    widthImg = img.shape[0]
    heightImg = img.shape[1]
    # Two sets of points src & dest
    pointset1 = np.float32(biggest)
    pointset2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    # calculate Transformation Matrix
    matrix = cv2.getPerspectiveTransform(pointset1, pointset2)
    # Transform the input image
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    # Crop the image
    imgCropped = imgOutput[20:imgOutput.shape[0]-20,20:imgOutput.shape[1]-20]
    #resize
    imgCropped = cv2.resize(imgCropped,(widthImg,heightImg))

    return imgCropped

def bg_transform (image:np.ndarray)-> np.ndarray:

    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
    comparison_array = imgGray < 150


    bg_image = np.zeros_like(image, dtype=np.uint8)

    for chanel in range(3):

        bg_image[:,:,chanel] = np.where(comparison_array , image[:,:,chanel], 255)


    return bg_image



if __name__ == "__main__":

    path = "/home/vahid/Development/Python/Scanner/input_image/IMG_6412.jpg"
    img = cv2.imread(path)
    input_image = img.copy()
    process_image = preProcessing(img)
    document_counter = getContours(process_image)
    myPoints = document_counter.reshape(document_counter.shape[0], 2)
    # get 4 important points
    f_points = get_corners(myPoints)
    # visuallize them
    f_points = f_points.reshape(4,2)
    for x,y in f_points:
        img = cv2.circle(img, (x,y), 7, (0, 0, 255), 7)

    # visuallize the lines
    img = cv2.line(img, tuple(f_points[0,:]), tuple(f_points[1,:]), (0, 255, 0), 5)
    img = cv2.line(img, tuple(f_points[0, :]), tuple(f_points[2, :]), (0, 255, 0), 7)
    img = cv2.line(img, tuple(f_points[1, :]), tuple(f_points[3, :]), (0, 255, 0), 7)
    img = cv2.line(img, tuple(f_points[2, :]), tuple(f_points[3, :]), (0, 255, 0), 7)


    Outimage = Transform(img, f_points)
    new_bg_image = bg_transform (Outimage)
    filename_output = '/home/vahid/Development/Python/Scanner/output_image/Image_origianl.jpg'
    filename_newbg = '/home/vahid/Development/Python/Scanner/output_image/Image_newbg.jpg'

    Outimage = cv2.resize(Outimage, (2480, 3508))
    new_bg_image = cv2.resize(new_bg_image, (2480, 3508))

    cv2.imwrite(filename_output, Outimage)
    cv2.imwrite(filename_newbg, new_bg_image)
    txt = ['input_image', 'preprocess_image','counters','Output_image','new_bg_image']
    show_image_series('final',[input_image,process_image ,img ,Outimage,new_bg_image], txt)
    cv2.waitKey(0)
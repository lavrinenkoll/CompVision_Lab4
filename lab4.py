import cv2
import numpy as np

#set video path
video = cv2.VideoCapture("video.mp4")
#set video size
width, height = 600, 400
#set scale for video
skale_w, skale_h = width / 600, height / 400


#edit video color and filter
def video_edit(video):

    #read video
    pixels = video.read()[1]
    resize = cv2.resize(pixels, (width, height))
    show = resize.copy()

    #edit video color
    #convert to hsv
    hsv = cv2.cvtColor(resize, cv2.COLOR_BGR2HSV)
    alpha = 1
    beta = 0
    #change contrast
    new_hsv = cv2.addWeighted(hsv, alpha, hsv, beta, 0)
    #change brightness
    change_bright = correction_brightness(new_hsv)
    #set negative color for some part of image
    param = np.array([[round(0*skale_w), round(170*skale_h)], [round(0*skale_w), round(400*skale_h)],
                      [round(500*skale_w), round(400*skale_h)], [round(270*skale_w), round(170*skale_h)]])
    negative_color = negative(change_bright, param)
    edited_color = negative_color

    #edit video filter
    gray = cv2.cvtColor(edited_color, cv2.COLOR_RGB2GRAY)
    #blur all image
    blur = cv2.GaussianBlur(gray, (7, 7), 2)

    # blur buildings
    blured_part = blur[round(0*skale_h):round(200*skale_h), round(0*skale_w):round(600*skale_w)]
    blured_part = cv2.blur(blured_part, (15, 15))
    blur[round(0*skale_h):round(200*skale_h), round(0*skale_w):round(600*skale_w)] = blured_part

    # blur markings
    blured_part = blur[round(225*skale_h):round(265*skale_h), round(175*skale_w):round(340*skale_w)]
    blured_part = cv2.blur(blured_part, (5, 5))
    blur[round(225*skale_h):round(265*skale_h), round(175*skale_w):round(340*skale_w)] = blured_part

    # cannys filter
    canny = cv2.Canny(blur, 80, 100)
    # close rectangles
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
    edited_by_filter = closed

    return edited_color, edited_by_filter, show


#correction brightness for all image
def correction_brightness(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    image = cdf[image]
    return image


#set negative color for some part of image (param)
def negative(pixels, param):
    new = pixels.copy()
    for i in range(param[0][1], param[2][1]):
        for j in range(param[0][0], param[3][0]):
            r = 255 - pixels[i, j][0]
            g = 255 - pixels[i, j][1]
            b = 255 - pixels[i, j][2]
            new[i, j] = (r, g, b)

    k = 0
    for i in range(param[3][1], param[2][1]):
        for j in range(param[3][0], param[3][0]+k):
            r = 255 - pixels[i, j][0]
            g = 255 - pixels[i, j][1]
            b = 255 - pixels[i, j][2]
            new[i, j] = (r, g, b)
        k+=1

    return new


#check if number plate inside car
def is_number_plate_inside_car(number_plate_contour, car_contour):
    number_plate_rectangle = cv2.boundingRect(number_plate_contour)
    car_rectangle = cv2.boundingRect(car_contour)

    return (number_plate_rectangle[0] >= car_rectangle[0] and
            number_plate_rectangle[1] >= car_rectangle[1] and
            number_plate_rectangle[0] + number_plate_rectangle[2] <= car_rectangle[0] + car_rectangle[2] and
            number_plate_rectangle[1] + number_plate_rectangle[3] <= car_rectangle[1] + car_rectangle[3])


#main loop for video
while True:
    # get edited video
    edit1, edit2, show = video_edit(video)
    contours, hierarchy = cv2.findContours(edit2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cars = []
    # find contours, check if it is human, car or number plate and draw it
    for contour in contours:
        # get rectangle around contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.001 * peri, True)
        rectangle = cv2.boundingRect(approx)
        area_rectangle = rectangle[2] * rectangle[3]

        # check if it is human, car or number plate
        if (rectangle[3]/rectangle[2]>2 and rectangle[3]/rectangle[2]<3 and area_rectangle>900*skale_h*skale_w
                and area_rectangle<1500*skale_h*skale_w):
            cv2.rectangle(show, (rectangle[0], rectangle[1]),
                          (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]), (0, 255, 0), 1)
            cv2.putText(show, "Human", (rectangle[0], rectangle[1]), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0))
            cv2.drawContours(show, [approx], -1, (0, 255, 0), 1)

        elif (rectangle[2]/rectangle[3]>1.5 and rectangle[2]/rectangle[3]<3 and area_rectangle>1500*skale_h*skale_w
              and area_rectangle<3000*skale_h*skale_w):
            cv2.rectangle(show, (rectangle[0], rectangle[1]),
                          (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]), (0, 0, 255), 1)
            cv2.putText(show, "Car", (rectangle[0], rectangle[1]), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 0, 255))
            cv2.drawContours(show, [approx], -1, (0, 0, 255), 1)
            cars.append(approx)

        elif (rectangle[2]/rectangle[3]>3 and rectangle[2]/rectangle[3]<6 and area_rectangle<300*skale_h*skale_w
              and area_rectangle>200*skale_h*skale_w):
            for car in cars:
                if is_number_plate_inside_car(approx, car):
                    cv2.rectangle(show, (rectangle[0], rectangle[1]),
                                  (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]), (255, 0, 0), 1)
                    cv2.putText(show, "Number", (rectangle[0], rectangle[1]), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 0, 0))
                    cv2.drawContours(show, [approx], -1, (255, 0, 0), 1)
                    break
    # show video
    cv2.imshow("Video with contours", show)
    cv2.imshow("Video with color edit", edit1)
    cv2.imshow("Video with filter edit", edit2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
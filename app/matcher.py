import utils
import cv2


def multiscale_template_matching(pic, template, method, scale_range,mask=None):
    (tH, tW) = template.shape[:2]
    res = []

    # loop over scales
    for scale in scale_range:
        # resize acc. to the scale
        resized = utils.resize(pic, width=int(pic.shape[1] * scale))

        # tracking ratio
        r = pic.shape[1] / float(resized.shape[1])

        #break if the resized image is smaller than the template
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        # match template and append result
        result = cv2.matchTemplate(resized, template, method, mask = mask)
        res.append((cv2.minMaxLoc(result), r, scale,(tH, tW)))

    return max(res or [[0,0,0,0],0,0,[0,0]], key=lambda x: x[0][1] )

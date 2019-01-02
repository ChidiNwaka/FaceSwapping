# Test for video camera

import cv2

def resize_to_fit(image, target):

    image_h, image_w, _ = image.shape
    target_h, target_w, _ = target.shape

    factor = min(target_h / image_h, target_w / image_w)
    new_image_h = int(factor * image_h)
    new_image_w = int(factor * image_w)
    new_image_shape = (new_image_w, new_image_h)

    return cv2.resize(image, new_image_shape)


def apply_face(face, target):
    face_h, face_w, _ = face.shape

    target_with_face = target.copy() # This duplicates the target image.
    target_with_face[:face_h, :face_w] = face

    return target_with_face

def face_swap(face1, face2):
    resized_face1 = resize_to_fit(face1, face2)
    resized_face2 = resize_to_fit(face2, face1)

    swapped_face1 = apply_face(resized_face2, face1)
    swapped_face2 = apply_face(resized_face1, face2)

    return swapped_face1, swapped_face2

def get_pairs(iterable):
    iterator = iter(iterable)
    return zip(iterator, iterator)


def main():
    cap = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier("parameters.xml")

    while True:
        ret, frame = cap.read()
        rectangles = cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=4,
                                              minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for ((x, y, w, h), (x2, y2, w2, h2)) in get_pairs(rectangles):
            # cv2.rectangle(frame, (x, y), (x + w,  y + h), (0, 255, 0), 2)
            face1 = (slice(y, y + h), slice(x, x + w))
            face2 = (slice(y2, y2 + h2), slice(x2, x2 + w2))

            frame[face1], frame[face2] = face_swap(frame[face1], frame[face2])


        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__=='__main__':
    main()
from image_processor import ImageProcess
import find_labels as fl
import label_generator as lg
import text_processor as tp
if __name__ == '__main__':
    img_name = input("image name: ")
    flag = input("preprocess image?(y/n):") == 'y'
    processor = ImageProcess('images/' + img_name, flag)
    print(label_from_text)
    # plots preprocessed image
    processor.plot_preprocessed_image()
    # detects objects in preprocessed image
    candidates = processor.get_candidates()
    # plots objects detected
    processor.plot_to_check(candidates, 'Total Objects Detected')
    # selects objects containing text
    text = processor.predict_char()
    # plots the realigned text
    raw_result = processor.realign_text()


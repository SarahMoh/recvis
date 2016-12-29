import numpy as np
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

imgsDir = '/media/evann/Data/MS COCO/train2014'
featuresDir = 'Features/'
vecImages = np.load(featuresDir + 'imgFeatures.npy').item()
annotations = np.load(featuresDir + 'imgAnnotations.npy').item()
vecWords = np.load(featuresDir + 'wordFeatures.npy').item()
freq = np.load(featuresDir + 'wordFrequencies.npy').item()
stopWordsEn = stopwords.words("english")

dictionnary = vecWords.keys()
vecWordsSize = len(vecWords[vecWords.keys()[0]])
vecImagesSize = len(vecImages[vecImages.keys()[0]][0])

withGensim = True
showImage = True


def sentenceToWords(sentence):
    words = sentence.lower().replace(',', '').replace('.', '').replace('"', '').split()
    words = [word for word in words if (word not in stopWordsEn) and word in dictionnary]
    return words


def sentenceToVec(sentence):
    words = sentenceToWords(sentence)
    sentVec = np.zeros_like(vecWords['word'])
    sumFreq = 0
    for word in words:
        weight = 1/freq[word]
        sentVec += vecWords[word] * weight
        sumFreq += weight
    sentVec /= sumFreq
    return sentVec


def test_words_features():
    print('%d word vectors loaded' % len(vecWords))
    print('Word vectors size : %d' % len(vecWords['love']))
    print('Frequence of the word "love": %f' % freq['love'])
    print('Vector for the word "love": %s' % vecWords['love'])


def test_images_features():
    print('\n%d image vectors loaded' % len(vecImages))
    print('Image vectors size : %d' % len(vecImages['000072'][0]))
    print(vecImages['000072'])


def test_annnotations():
    keys = annotations.keys()
    key = keys[np.random.randint(len(keys))]
    print("\nAnnotation for image %s:" % key)
    for ann in annotations[key]:
        print(ann)

    if (showImage):
        imgName = 'COCO_train2014_000000%s.jpg' % key
        I = plt.imread('%s/%s' % (imgsDir, imgName))
        plt.imshow(I)
        plt.show()


def test_gensim():
    sentence = 'A cat on a table'
    sentenceVector = sentenceToVec(sentence)

    if (withGensim):
        from gensim.models import Word2Vec
        model = Word2Vec.load(featuresDir + 'w2v_model')

        # Here is a test weighting words by their inverse frequency
        test = model.similar_by_vector(sentenceVector, topn=10)
        print(test)
        print('\n')

        # Here is a test weighting words all the same
        test = model.most_similar(positive=sentenceToWords(sentence), topn=10)
        print(test)
        print('\n')

    print('Cat freq : %f' % freq['cat'])
    print('Table freq : %f' % freq['table'])


def test_real_use():
    # Choose random image
    imageIds = annotations.keys()
    imageId = imageIds[np.random.randint(len(imageIds))]

    # Create a vector for this image's annotions by summing every words in each annotations
    vector = np.zeros(vecWordsSize)
    for ann in annotations[imageId]:
        vector += sentenceToVec(ann)

    annotationVectorForImage = vector
    imageVectorForImage = vecImages[imageId]

    print('\nAnnotation Vector : %s' % annotationVectorForImage)
    print('Image Vector : %s' % imageVectorForImage)

test_words_features()
test_images_features()
test_annnotations()
test_gensim()
test_real_use()

import sys
import argparse
import movie_classifier

def main(args):
    titl = args.title
    desc = args.description
    split = args.split
    classifier = args.classifier
    dataset = args.dataset

    if (split != None and classifier != None):
        print('Got input for both dataset split and Classifier to be used....Starting Traning...with training data')
        p = movie_classifier.Preprocess_Data(dataset)
        p.create_training_dataset()
        # print(type(split))
        t = movie_classifier.Training_Data(p.movies, float(split))
        f1 = t.Training_Initialization()

    elif (titl != None and desc != None):
        if (len(titl) > 2 and len(desc) > 2):
            print('Got title and Description of movie for finding Genre...Initiating Testing')
            tes = movie_classifier.Model_Testing(titl, desc)
            op = tes.pred()
            output_dict = {}
            output_dict = {'title': titl, 'description': desc, 'genre': op}
            print(output_dict)
        else:
            print(
                'Note: Title or Description field is left empty.... Kindly add valid Movie Title or Description to run the Algorithm')

    else:
        print('Kindly type movie_classifier -h to get help for input arguments to use the code')

if __name__ == "__main__":

    welcome = "Command Line based Genre Predictor"

    parser = argparse.ArgumentParser(description=welcome)
    parser.add_argument('--title', '-t',  type= str,
                        help='Movie name that you want to Know Genre of. i.e. "Harry Potter"')
    parser.add_argument('--description', '-d',  type= str,
                        help='the movie description')
    parser.add_argument('--split', '-s', type= float,
                        help='training dataset split')
    parser.add_argument('--classifier', '-c', type= str,
                        help='Classifier to be used')
    parser.add_argument('--dataset', '-ds', default='data/movies_metadata.csv',
                        help='training dataset to be used in csv format please eg: "movies_metadata.csv" from movie-lens dataset')
    #parser.parse_args()
    args = parser.parse_args()

    main(args)


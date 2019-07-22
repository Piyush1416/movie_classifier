import unittest

from movie_classifier import movie_classifier


class Test_Preprocessing(unittest.TestCase):

    def test_clean_text_overview(self):
        '''
        This test is used to check if data cleaning is done properly
        '''

        result = "crap sample text numbers money"
        data = "This is a Crap !!! , sample text with 123 NUMBERS and $123 money in it?"
        y = movie_classifier.Preprocess_Data.clean_text_overview(self,data)
        self.assertEqual(result, y)

class Test_Training_Data(unittest.TestCase):

    def test_F1_Score_after_Training(self):
        '''
        This test will be used to check how well the model performed by changing certain parameters
        like train-test spilt, classifers or dataset used for training compared to current models F1 score
        '''

        current_model_f1_score = 0.46548777769522254

        p = movie_classifier.Preprocess_Data("data/movies_metadata.csv")  # default file set...else uer specified file will come
        p.create_training_dataset()
        t = movie_classifier.Training_Data(p.movies, 0.2)
        f1 = t.Training_Initialization()
        self.assertTrue(current_model_f1_score <= f1)


class Test_Testing_Data(unittest.TestCase):

    def test_Genre_Prediction(self):
        '''
        This test will check how well the model predicted output
        '''

        Expected_Genre = "[('Horror',)]"
        tes = movie_classifier.Model_Testing('hollywood chainsaw hookers',
                            'l private eye hired worried mother find missing runaway daughter samantha private dick jack chandler searches whereabouts misfortune encountering evil cult worships egyptian god methods human sacrifice using chainsaws choice appeasing deity chandler learns samantha revenge store master bevy blood thirsty chainsaw wielding hookers')
        pre = tes.pred()
        self.assertTrue(pre,Expected_Genre)






if __name__ == '__main__':
    unittest.main()
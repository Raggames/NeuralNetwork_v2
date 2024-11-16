using Atom.MachineLearning.Core;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using static Atom.MachineLearning.Supervised.Recommender.ItemBased.ItemBasedRecommenderModel;

namespace Atom.MachineLearning.Supervised.Recommender.ItemBased
{
    public class IBRM_Test : MonoBehaviour
    {
        [SerializeField] private ItemBasedRecommenderModel _model;

        [Button]
        private async void Test_1(SimilarityFunctions similarityFunction)
        {
            _model = new ItemBasedRecommenderModel("ibrm-test", similarityFunction);

            // creating a small set of five user rating chocolate, coffee or tea
            // 0 mean no rating
            // ratings are from 1 to 5 included
            /*var test_set = new NVector[8];

            test_set[0] = new NVector(new double[] { 3f, 5f, 0f }); // didnt note chocolate, love coffee, like tea
            test_set[1] = new NVector(new double[] { 5f, 3f, 0f }); // like tea, chocolate not that much, didn't note coffee
            test_set[2] = new NVector(new double[] { 0f, 1f, 5f }); // love tea, like coffee, didn't note chocolate
            test_set[3] = new NVector(new double[] { 5f, 3.5f, 0f }); // love chocolate, like tea not so much, how will he like coffee?
            test_set[4] = new NVector(new double[] { 1f, 0f, 4.5f }); // love chocolate, didn't rate any other
            test_set[5] = new NVector(new double[] { 0.5f, 0f, 2.5f }); // low rater that prefer tea
            test_set[6] = new NVector(new double[] { 3.5f, 0f, 0f }); // low rater that prefer chocolate
            test_set[7] = new NVector(new double[] { 0f, 0f, 0f }); // didn't rate any*/

            var test_set = new NVector[4];
            test_set[0] = new NVector(new double[] { 3f, 5f, 0f });
            test_set[1] = new NVector(new double[] { 5f, 3f, 0f }); 
            test_set[2] = new NVector(new double[] { 1f, 3f, 5f }); 
            test_set[3] = new NVector(new double[] { 1f, 2f, 4f }); 
          

            var result = await _model.Fit(test_set);

            Debug.Log("Similarity matrix : ");
            for (int i = 0; i < _model.itemSimilarityMatrix.Rows; ++i)
            {
                var row_vec = new NVector(_model.itemSimilarityMatrix.Columns);
                for (int j = 0; j < _model.itemSimilarityMatrix.Columns; ++j)
                {
                    row_vec[j] = _model.itemSimilarityMatrix[i, j];
                }

                Debug.Log($"Item {i} > " + row_vec.ToString());
            }

            Debug.Log("***************");
            Debug.Log("Prediction completed ratings");

            // now compute missing scores
            var prediction_completed_ratings = new NVector[test_set.Length];
            for (int i = 0; i < test_set.Length; i++)
            {
                prediction_completed_ratings[i] = _model.Predict(test_set[i]);
                Debug.Log($"User {i} > " + prediction_completed_ratings[i].ToString());
            }


        }
    }
}

using Atom.MachineLearning.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Unsupervised.SelfOrganizingMap
{
    /// <summary>
    /// Self Organizing Map (or Kohonen Map or SOM) is a type of Artificial Neural Network which is also inspired by biological models of neural systems from the 1970s. 
    /// It follows an unsupervised learning approach and trained its network through a competitive learning algorithm. 
    /// SOM is used for clustering and mapping (or dimensionality reduction) techniques to map multidimensional data onto lower-dimensional which allows people to reduce complex problems for easy interpretation. 
    /// SOM has two layers, one is the Input layer and the other one is the Output layer. 
    /// </summary>
    public class SOMModel : IMLModel<NVector, KohonenMatchingUnit>
    {
        public enum DistanceFunctions
        {
            Euclidian,
            Manhattan,
            Minkowski3,
        }

        public string ModelName { get; set; }
        public string ModelVersion { get; set; }

        /// <summary>
        /// SOM was introduced by Finnish professor Teuvo Kohonen in the 1980s
        /// The map is a 2D grid representation of the input features iteratively computed during the training process
        /// 3rd dimension is the weight vector of each neuron of the 2D grid
        /// </summary>
        [LearnedParameter] private NVector[,] _kohonenMap;

        public NVector[,] kohonenMap => _kohonenMap;

        private Func<NVector, NVector, double> _distanceFunction;

        public SOMModel(Func<NVector, NVector, double> distanceFunction = null)
        {
            if (distanceFunction == null)
                _distanceFunction = (a, b) => NVector.Euclidian(a, b);
            else
                _distanceFunction = distanceFunction;
        }

        public SOMModel(DistanceFunctions distanceFunction)
        {
            switch (distanceFunction)
            {
                case DistanceFunctions.Euclidian:
                    _distanceFunction = (a, b) => NVector.Euclidian(a, b);
                    break;
                case DistanceFunctions.Manhattan:
                    _distanceFunction = (a, b) => NVector.Manhattan(a, b);
                    break;
                case DistanceFunctions.Minkowski3:
                    _distanceFunction = (a, b) => NVector.Mnkowski(a, b, 3);
                    break;
            }
        }

        public void InitializeMap(int nodesCount, int featureDimensions)
        {
            var matrix1Dsize = (int)Math.Round(Math.Sqrt(nodesCount));
            _kohonenMap = new NVector[matrix1Dsize, matrix1Dsize];

            // set random weights in the direction of the dataset
            for (int x = 0; x < _kohonenMap.GetLength(0); ++x)
            {
                for (int y = 0; y < _kohonenMap.GetLength(1); ++y)
                {
                    _kohonenMap[x, y] = new NVector(featureDimensions).Random(.001, .01);
                }
            }
        }

        /// <summary>
        /// Set the weight of the neuron at given coordinates
        /// </summary>
        /// <param name="x_coor"></param>
        /// <param name="y_coor"></param>
        /// <param name="weight"></param>
        public void UpdateWeight(int x_coor, int y_coor, NVector weight)
        {
            _kohonenMap[x_coor, y_coor] = weight;
        }

        /// <summary>
        /// Prediction will return the bestmatching unit of the current kohonen map state
        /// </summary>
        /// <param name="inputData"></param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        public KohonenMatchingUnit Predict(NVector inputData)
        {
            double best_distance = double.MaxValue;
            KohonenMatchingUnit best_matching_unit = new KohonenMatchingUnit();

           /* for (int x = 0; x < _kohonenMap.GetLength(0); ++x)
            {
                for (int y = 0; y < _kohonenMap.GetLength(1); ++y)
                {
                    var matching_unit_vector = _kohonenMap[x, y];
                    var dist = _distanceFunction(matching_unit_vector, inputData);

                    if (dist < best_distance)
                    {
                        best_distance = dist;
                        best_matching_unit = new KohonenMatchingUnit()
                        {
                            WeightVector = matching_unit_vector,
                            XCoordinate = x,
                            YCoordinate = y,
                            Distance = 0,
                        };
                    }
                }
            }
*/
            Parallel.For(0, _kohonenMap.GetLength(0), (x) =>
            {
                for (int y = 0; y < _kohonenMap.GetLength(1); ++y)
                {
                    var matching_unit_vector = _kohonenMap[x, y];
                    var dist = _distanceFunction(matching_unit_vector, inputData);

                    if (dist < best_distance)
                    {
                        best_distance = dist;
                        best_matching_unit = new KohonenMatchingUnit()
                        {
                            WeightVector = matching_unit_vector,
                            XCoordinate = x,
                            YCoordinate = y,
                            Distance = dist,
                        };
                    }
                }
            });
                   
            return best_matching_unit;
        }

    public List<KohonenMatchingUnit> GetNeighboors(int xOrigin, int yOrigin, double distance)
    {
        var results = new List<KohonenMatchingUnit>();

        for (int x = 0; x < _kohonenMap.GetLength(0); ++x)
        {
            for (int y = 0; y < _kohonenMap.GetLength(1); ++y)
            {
                // grid-based neihboor using euclidian
                var grid_euclidian_dist = Euclidian(xOrigin, yOrigin, x, y);
                if (grid_euclidian_dist < distance)
                    results.Add(new KohonenMatchingUnit()
                    {
                        WeightVector = _kohonenMap[x, y],
                        XCoordinate = x,
                        YCoordinate = y,
                        Distance = grid_euclidian_dist,
                    });
            }
        }

        return results;
    }

    private double Euclidian(int xOrigin, int yOrigin, int x, int y)
    {
        var dist = Math.Pow(xOrigin - x, 2);
        dist += Math.Pow(yOrigin - y, 2);

        return Math.Sqrt(dist);
    }
}
}

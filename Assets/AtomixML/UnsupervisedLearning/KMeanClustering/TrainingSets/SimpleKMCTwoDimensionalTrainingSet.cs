using Atom.MachineLearning.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Unsupervised.KMeanClustering
{
    public class SimpleKMCTwoDimensionalTrainingSet : KMCTrainingSet
    {
        private VectorNInputData[] _features;
        public override VectorNInputData[] Features
        {
            get
            {
                if(_features == null)
                {

                    // Generate the dataset
                    int rows = 200;
                    int columns = 2;
                    double[,] datas = new double[rows, columns];

                    Random random = new Random();


                    for (int i = 0; i < rows; i++)
                    {
                        datas[i, 0] = random.NextDouble() * 10; // x values between 0 and 10
                        datas[i, 1] = random.NextDouble() * 10; // y values between 0 and 10

                        if (i < rows / 4) // cluster 0: top left corner
                            datas[i, 0] += 2;
                        else if (i < rows / 2) // cluster 1: center of the square
                            datas[i, 0] = 5 + random.NextDouble() * 3; // x values between 2 and 8
                        else if (i < 3 * rows / 4) // cluster 2: bottom left corner
                            datas[i, 0] -= 2;
                        else // cluster 3: top right corner
                            datas[i, 1] = 10 + random.NextDouble() * 5; // y values between 5 and 15

                        if (i < rows / 4) // cluster 1: bottom left corner
                            datas[i, 1] += 2;
                        else if (i < rows / 2) // cluster 2: center of the square
                            datas[i, 1] = 5 + random.NextDouble() * 3; // y values between 2 and 8
                        else if (i < 3 * rows / 4) // cluster 3: top left corner
                            datas[i, 1] -= 2;
                    }

                    _features = new VectorNInputData[datas.GetLength(0)];

                    for (int i = 0; i < datas.GetLength(0); ++i)
                        _features[i] = new VectorNInputData(datas[i, 0], datas[i, 1]);

                }

                return _features;
            }
        }

/*        public double[,] ___datas = new double[,] {
            { -1.294858, -1.035934 },
            { -2.542902, -1.386228 },
            { -1.333254, -2.489007 },
            { -0.742485, -1.242989 },
            { -0.689398, -0.938147 },
            { -1.258111, -1.245493 },
            { -1.629322, -0.809516 },
            { -0.753073, -1.320548 },
            { -1.679330, -0.943314 },
            { -1.399613, -1.193334 },
            { -1.614390, -0.933303 },
            { -1.167730, -1.173366 },
            { -1.388622, -1.230234 },
            { -1.482779, -1.142043 },
            { -0.680899, -1.395434 },
            { -1.397192, -1.503032 },
            { -1.303012, -0.974373 },
            { -1.208231, -0.915784 },
            { -1.097637, -0.995233 },
            { -0.886732, -1.436202 },
            { -1.216303, -1.081222 },
            { -1.216303, -0.837732 },
            { -0.901328, -1.256342 },
            { -1.597123, -0.912473 },
            { -1.316829, -1.217403 },
        };
*/
    }
}

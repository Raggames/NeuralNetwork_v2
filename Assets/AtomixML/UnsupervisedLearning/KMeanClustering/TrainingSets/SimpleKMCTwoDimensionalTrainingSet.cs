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
        public double[,] datas = new double[,] {
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

        public override VectorNInputData[] Features
        {
            get
            {
                var features = new VectorNInputData[datas.Length];

                for (int i = 0; i < datas.Length; ++i)
                    features[i] = new VectorNInputData(datas[i, 0], datas[i, 1]);

                return features;
            }
        }

    }
}

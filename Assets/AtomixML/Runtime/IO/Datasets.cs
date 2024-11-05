using Atom.MachineLearning.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.IO
{
    /// <summary>
    /// A bunch of method to return test datasets from the project
    /// </summary>
    public static class Datasets
    {
        public static string[,] Flowers_All()
        {
            return DatasetReader.ReadCSV("Assets/AtomixML/Resources/Datasets/flowers/iris.data.txt", ',', 0);
        }

        public static string[,] Housing_Train()
        {
            return DatasetReader.ReadCSV("Assets/AtomixML/Resources/Datasets/housing/df_train.txt", ',', 1);
        }

        public static string[,] Housing_Test()
        {
            return DatasetReader.ReadCSV("Assets/AtomixML/Resources/Datasets/housing/df_test.txt", ',', 1);
        }

        public static NVector[] Mnist_8x8_Vectorized_All()
        {
            var textures = DatasetReader.ReadTextures("Datasets/mnist");

            var vectors_array = new NVector[textures.Count];
            for (int i = 0; i < textures.Count; ++i)
            {
                var matrix = TransformationUtils.Texture2DToMatrix(textures[i]);
                var pooled = TransformationUtils.PoolAverage(matrix, 4, 2);
                var array = TransformationUtils.MatrixToArray(pooled);
                vectors_array[i] = new NVector(array);
            }

            return vectors_array;
        }
    }
}

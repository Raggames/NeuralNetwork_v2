using Atom.MachineLearning.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

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

        public static Texture2D[] Mnist_8x8_TexturePooled_All()
        {
            var textures = DatasetReader.ReadTextures("Datasets/mnist");

            var vectors_array = new Texture2D[textures.Count];
            for (int i = 0; i < textures.Count; ++i)
            {
                var matrix = TransformationUtils.Texture2DToMatrix(textures[i]);
                var pooled = TransformationUtils.PoolAverage(matrix, 4, 2);
                var array = TransformationUtils.MatrixToTexture2D(pooled);
                vectors_array[i] = array;
            }

            return vectors_array;
        }

        public static NVector[] Rnd_bw_2x2_Vectorized_All()
        {
            var textures = DatasetReader.ReadTextures("Datasets/rnd_bw");

            var vectors_array = new NVector[textures.Count];
            for (int i = 0; i < textures.Count; ++i)
            {
                var array = TransformationUtils.Texture2DToArray(textures[i]);
                vectors_array[i] = new NVector(array);
            }

            return vectors_array;
        }

        public static NVector[] Rnd_bw_8x8_Vectorized_All()
        {
            var textures = DatasetReader.ReadTextures("Datasets/rnd_bw_10x10");

            var vectors_array = new NVector[textures.Count];
            for (int i = 0; i < textures.Count; ++i)
            {
                var array = TransformationUtils.Texture2DToArray(textures[i]);
                vectors_array[i] = new NVector(array);
            }

            return vectors_array;
        }

        public static Texture2D[] Rnd_bw_2x2_Texture_All()
        {
            return DatasetReader.ReadTextures("Datasets/rnd_bw").ToArray();
        }
    }
}

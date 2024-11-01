using Atom.MachineLearning.IO;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace NeuralNetwork
{
    [Serializable]
    public struct TextureSet
    {
        public Texture2D[] set;
    }
    [CreateAssetMenu(menuName = "TrainingSets/MNISTTrainingSetting")]
    public class MNISTTrainingSetting : TrainingSettingBase
    {
        public List<TextureSet> TraininngSetTextures;

        public Texture2D[] TestSetTextures;

        public double[][,] X_Datas;
        public double[][] T_Datas;

        public override void Init()
        {
            GenerateMatrices();
        }

        private void GenerateMatrices()
        {
            X_Datas = new double[TraininngSetTextures.Count * 60][,];
            T_Datas = new double[TraininngSetTextures.Count * 60][];
            int indexer = 0;

            for (int s = 0; s < TraininngSetTextures.Count; ++s)
            {
                double[][,] image_x_values = new double[TraininngSetTextures[s].set.Length][,];
                double[][] image_t_values = new double[TraininngSetTextures[s].set.Length][];

                for (int t = 0; t < TraininngSetTextures[s].set.Length; ++t)
                {
                    image_x_values[t] = new double[TraininngSetTextures[s].set[t].width, TraininngSetTextures[s].set[t].height];
                    image_t_values[t] = new double[10];

                    for (int i = 0; i < TraininngSetTextures[s].set[t].width; ++i)
                    {
                        for (int j = 0; j < TraininngSetTextures[s].set[t].height; ++j)
                        {
                            var pix = TraininngSetTextures[s].set[t].GetPixel(i, j);
                            float value = ((pix.r + pix.g + pix.b) / 3f) * pix.a;
                            image_x_values[t][i, j] = value;
                        }
                    }

                    for(int i = 0; i < 10; ++i)
                    {
                        if(s == i)
                        {
                            image_t_values[t][i] = 1;

                        }
                        else
                        {
                            image_t_values[t][i] = 0;
                        }
                    }
                }

                for(int i = 0; i < image_x_values.Length; ++i)
                {
                    X_Datas[indexer] = image_x_values[i];
                    T_Datas[indexer] = image_t_values[i];
                    indexer++;
                }                
            }
        }

        public override void GetTrainDatas(out double[][] x_datas, out double[][] t_datas)
        {
            x_datas = X_Datas.Select(t => VectorizationUtils.MatrixToArray(t)).ToArray();
            t_datas = T_Datas;
        }

        public override void GetMatrixTrainDatas(out double[][,] x_datas, out double[][] t_datas)
        {
            x_datas = X_Datas;
            t_datas = T_Datas;
        }

        public override bool ValidateRun(double[] y_val, double[] t_val)
        {
            int index = NeuralNetworkMathHelper.MaxIndex(y_val);
            int tMaxIndex = NeuralNetworkMathHelper.MaxIndex(t_val);
            if (index.Equals(tMaxIndex))
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    }
}

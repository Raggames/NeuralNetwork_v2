using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace NeuralNetwork
{
    [CreateAssetMenu(menuName = "TrainingSets/MultilanguageClassificationTrainingSetting")]
    class MultilanguageClassificationTrainingSetting : RealFalseWordRecognitionTrainingSetting
    {
        protected string[] data_en;
        protected string[] data_de;
        protected string[] data_es;

        protected List<string[]> all_datas = new List<string[]>();

        public override void Init()
        {
            data_fr = new WordReader("fr").Read();
            data_de = new WordReader("de").Read();
            data_en = new WordReader("en").Read();
            data_es = new WordReader("es").Read();

            all_datas.Add(data_fr);
            all_datas.Add(data_de);
            all_datas.Add(data_en);
            all_datas.Add(data_es);
        }


        public override void GetTrainDatas(out double[][] x_datas, out double[][] t_datas)
        {
            x_datas = new double[TrainingDataLenght][];
            t_datas = new double[TrainingDataLenght][];

            for(int i = 0; i < x_datas.GetLength(0); ++i)
            {
                GetNextValues(out x_datas[i], out t_datas[i]);
            }
        }

        public override void GetNextValues(out double[] x_val, out double[] t_val)
        {
            t_val = new double[all_datas.Count];

            // Over Ratio, real world
            int languageIndex = UnityEngine.Random.Range(0, all_datas.Count);

            for (int i = 0; i < all_datas.Count; ++i)
            {
                if (i == languageIndex)
                {
                    t_val[i] = 1;
                }
                else
                {
                    t_val[i] = 0;
                }
            }

            GetInputForData(all_datas[languageIndex]);

            x_val = GetDataArrayFromWord(input_debug);

            if (IS_DEBUG)
            {
                Debug.LogError(input_debug + " < = > " + UnwrapWord(x_val) + " " + languageIndex);
            }
        }

        /*        public override void GetNextValues(out double[] x_val, out double[] t_val)
                {
                    t_val = new double[5];

                    // Over Ratio, real world
                    if (UnityEngine.Random.Range(0, 100) > Real_Invented_Word_Ratio_Purcentage)
                    {
                        int languageIndex = UnityEngine.Random.Range(0, 4);

                        t_val[0] = 0; // False word false
                        for (int i = 0; i < 4; ++i)
                        {
                            if (i == languageIndex)
                            {
                                t_val[i + 1] = 1;
                            }
                            else
                            {
                                t_val[i + 1] = 0;
                            }
                        }

                        GetInputForData(all_datas[languageIndex]);

                        x_val = GetDataArrayFromWord(input_debug);

                        if (IS_DEBUG)
                        {
                            Debug.LogError(input_debug + " < = > " + UnwrapWord(x_val) + " " + languageIndex);
                        }

                    }
                    // Else invented one
                    else
                    {
                        int invented_word_lenght = UnityEngine.Random.Range(0, MaxWordLenght);
                        int[] invented_word_letters = new int[MaxWordLenght];
                        for (int i = 0; i < MaxWordLenght; ++i)
                        {
                            if (i < invented_word_lenght)
                            {
                                // 30 is hardcoded as the max possible known letter
                                invented_word_letters[i] += UnityEngine.Random.Range(0, MaxCharToIntValue);

                            }
                            else
                            {
                                invented_word_letters[i] = -1;
                            }
                        }

                        if (IS_DEBUG)
                        {
                            input_debug = GetWordByIntArray(invented_word_letters);
                        }

                        x_val = NormalizeWord(invented_word_letters);

                        t_val[0] = 1; // False word true
                        for (int i = 1; i < 5; ++i)
                        {
                            t_val[i] = 0;
                        }
                    }
                }
        */
        public override bool ValidateRun(double[] y_values, double[] t_values)
        {
            int index = NeuralNetworkMathHelper.MaxIndex(y_values);
            int tMaxIndex = NeuralNetworkMathHelper.MaxIndex(t_values);
            if (index.Equals(tMaxIndex))
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        protected void GetInputForData(string[] data)
        {
            input_debug = "";
            do
            {
                input_debug = data[UnityEngine.Random.Range(0, data.Length)];
            }
            while (input_debug.Length > MaxWordLenght);
        }
    }
}

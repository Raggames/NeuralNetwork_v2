using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Atom.MachineLearning.NeuralNetwork
{
    [CreateAssetMenu(menuName = "TrainingSets/RealFalseWordRecognitionTraining")]
    public class RealFalseWordRecognitionTrainingSetting : TrainingSettingBase
    {
        public int Real_Invented_Word_Ratio_Purcentage = 50;
        public int MaxWordLenght = 12;
        public float ValidationThreshold = .1f;
        public bool IS_DEBUG = false;
        protected string input_debug;

        public int MaxCharToIntValue = 0;
        public int MinCharToIntValue = 0;


        protected string[] data_fr;

        public override void Init()
        {
            data_fr = new WordReader("fr").Read();
        }

        protected double[] NormalizeWord(int[] word_characters_array)
        {
            double[] result = new double[word_characters_array.Length];

            for (int i = 0; i < word_characters_array.Length; ++i)
            {
                result[i] = NeuralNetworkMathHelper.Map(word_characters_array[i], MinCharToIntValue, (float)MaxCharToIntValue, 0, 1);
            }

            return result;
        }

        public string UnwrapWord(double[] input)
        {
            int[] wordarray = new int[input.Length];
            for (int i = 0; i < input.Length; ++i)
            {
                float remaped_value = NeuralNetworkMathHelper.Map((float)input[i], 0, 1, MinCharToIntValue, (float)MaxCharToIntValue);
                int int_remaped = Mathf.RoundToInt(remaped_value);
                wordarray[i] = int_remaped;
            }

            return GetWordByIntArray(wordarray);
        }

        protected float[] ConvertWordToFloatArray(string word)
        {
            float[] word_array = new float[word.Length];
            for (int i = 0; i < word.Length; ++i)
            {
                word_array[i] = NeuralNetworkMathHelper.Map(GetLetterIndex(word[i]), MinCharToIntValue, (float)MaxCharToIntValue, 0, 1);
            }
            return word_array;
        }

        protected double[] GetDataArrayFromWord(string word)
        {
            double[] word_array = new double[MaxWordLenght];
            for (int i = 0; i < MaxWordLenght; ++i)
            {
                if (i < word.Length)
                {
                    word_array[i] = NeuralNetworkMathHelper.Map(GetLetterIndex(word[i]), MinCharToIntValue, MaxCharToIntValue, 0, 1);
                }
                else
                {
                    word_array[i] = -1;
                }
            }

            return word_array;
        }

        protected int GetLetterIndex(char character)
        {
            int val1 = character - '0';
            MaxCharToIntValue = Math.Max(MaxCharToIntValue, val1);
            MinCharToIntValue = Math.Min(MinCharToIntValue, val1);

            return val1;
/*
            switch (character)
            {
                case 'a':
                    return 0;
                case 'b':
                    return 1;
                case 'c':
                    return 2;
                case 'd':
                    return 3;
                case 'e':
                    return 4;
                case 'f':
                    return 5;
                case 'g':
                    return 6;
                case 'h':
                    return 7;
                case 'i':
                    return 8;
                case 'j':
                    return 9;
                case 'k':
                    return 10;
                case 'l':
                    return 11;
                case 'm':
                    return 12;
                case 'n':
                    return 13;
                case 'o':
                    return 14;
                case 'p':
                    return 15;
                case 'q':
                    return 16;
                case 'r':
                    return 17;
                case 's':
                    return 18;
                case 't':
                    return 19;
                case 'u':
                    return 20;
                case 'v':
                    return 21;
                case 'w':
                    return 22;
                case 'x':
                    return 23;
                case 'y':
                    return 24;
                case 'z':
                    return 25;
                case 'é':
                    return 26;
                case 'è':
                    return 27;
                case 'à':
                    return 28;
                case 'ç':
                    return 29;
                case ' ':
                    return 30;
                default:
                    break;
            }

            return 30;*/
        }

        protected static string GetLetter(int index)
        {
            return ((char)(index + (int)'0')).ToString();

            /*switch (index)
            {
                case 30:
                    return " ";
                case 0:
                    return "a";
                case 1:
                    return "b";
                case 2:
                    return "c";
                case 3:
                    return "d";
                case 4:
                    return "e";
                case 5:
                    return "f";
                case 6:
                    return "g";
                case 7:
                    return "h";
                case 8:
                    return "i";
                case 9:
                    return "j";
                case 10:
                    return "k";
                case 11:
                    return "l";
                case 12:
                    return "m";
                case 13:
                    return "n";
                case 14:
                    return "o";
                case 15:
                    return "p";
                case 16:
                    return "q";
                case 17:
                    return "r";
                case 18:
                    return "s";
                case 19:
                    return "t";
                case 20:
                    return "u";
                case 21:
                    return "v";
                case 22:
                    return "w";
                case 23:
                    return "x";
                case 24:
                    return "y";
                case 25:
                    return "z";
                case 26:
                    return "é";
                case 27:
                    return "è";
                case 28:
                    return "à";
                case 29:
                    return "ç";
            }

            return "_";*/
        }

        protected static string GetWordByIntArray(int[] input)
        {
            string result = "";
            for (int i = 0; i < input.Length; ++i)
            {
                result += GetLetter(input[i]);
            }
            return result;
        }

        public override bool ValidateRun(double[] y_val, double[] t_val)
        {          
            if (Mathf.Abs((float)t_val[0] - (float)y_val[0]) < ValidationThreshold)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        public override void GetNextValues(out double[] x_val, out double[] t_val)
        {
            t_val = new double[1];

            // Over Ratio, real world
            if (UnityEngine.Random.Range(0, 100) > Real_Invented_Word_Ratio_Purcentage)
            {
                input_debug = "";
                do
                {
                    input_debug = data_fr[UnityEngine.Random.Range(0, data_fr.Length)];
                }
                while (input_debug.Length > MaxWordLenght);

                double[] vals =
                x_val = GetDataArrayFromWord(input_debug);

                if (IS_DEBUG)
                {
                    Debug.LogError(input_debug + " < = > " + UnwrapWord(x_val));
                }

                t_val[0] = 1;
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
                t_val[0] = 0;
            }
        }

        public override void GetTrainDatas(out double[][] x_datas, out double[][] t_datas)
        {
            x_datas = new double[TrainingDataLenght][];
            t_datas = new double[TrainingDataLenght][];

            for (int i = 0; i < x_datas.GetLength(0); ++i)
            {
                GetNextValues(out x_datas[i], out t_datas[i]);
            }
        }
    }
}


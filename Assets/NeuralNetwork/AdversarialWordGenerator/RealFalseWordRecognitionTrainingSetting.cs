using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NeuralNetwork
{
    [CreateAssetMenu(menuName = "TrainingSets/RealFalseWordRecognitionTraining")]
    public class RealFalseWordRecognitionTrainingSetting : TrainingSettingBase
    {
        public int Real_Invented_Word_Ratio_Purcentage = 50;
        public int MaxWordLenght = 12;
        public float ValidationThreshold = .1f;
        public bool IS_DEBUG = false;

        private string[] data;
        public override void Init()
        {
            data = new WordReader("fr").Read();
        }

        private int[] ConvertWordToIntArray(string word)
        {
            int[] word_array = new int[word.Length];
            for(int i = 0; i < word.Length; ++i)
            {
                word_array[i] = GetLetterIndex(word[i]);
            }
            return word_array;
        }

        private double[] NormalizeWord(int[] word_characters_array)
        {
            double[] result = new double[word_characters_array.Length];

            for(int i = 0; i < word_characters_array.Length; ++i)
            {
                result[i] = NeuralNetworkMathHelper.Norm01(0, 30, word_characters_array[i]);
            }

            return result;
        }

        private float[] ConvertWordToFloatArray(string word)
        {
            float[] word_array = new float[word.Length];
            for (int i = 0; i < word.Length; ++i)
            {
                word_array[i] = NeuralNetworkMathHelper.Norm01(0, 30, GetLetterIndex(word[i]));
            }
            return word_array;
        }

        private double[] GetDataArrayFromWord(string word)
        {
            double[] word_array = new double[MaxWordLenght];
            for (int i = 0; i < MaxWordLenght; ++i)
            {
                if(i< word.Length)
                {
                    word_array[i] = NeuralNetworkMathHelper.Norm01(0, 30, GetLetterIndex(word[i]));
                }
                else
                {
                    word_array[i] = -1;
                }
            }

            return word_array;
        }

        private int GetLetterIndex(char character)
        {
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
                default:
                    break;
            }

            return 30;
        }

        private string GetLetter(int index)
        {
            switch (index)
            {
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

            return "_";
        }

        private string GetWordByIntArray(int[] input)
        {
            string result = "";
            for(int i = 0; i < input.Length; ++i)
            {
                result += GetLetter(input[i]);
            }
            return result;
        }

        public override bool ValidateRun(double[] y_val, double[] t_val)
        {
            if (IS_DEBUG)
            {
                Debug.LogError("Input was => " + input_debug + " Result => " + y_val[0]);
            }

            if (Mathf.Abs((float)t_val[0] - (float)y_val[0]) < ValidationThreshold)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        private string input_debug;

        public override void GetNextValues(out double[] x_val, out double[] t_val)
        {
            t_val = new double[1];

            // Over Ratio, real world
            if (UnityEngine.Random.Range(0, 100) > Real_Invented_Word_Ratio_Purcentage)
            {
                input_debug = "";
                do
                {
                    input_debug = data[UnityEngine.Random.Range(0, data.Length)];
                }
                while (input_debug.Length > MaxWordLenght);

                double[] vals = 
                x_val = GetDataArrayFromWord(input_debug);
                t_val[0] = 1;
            }
            // Else invented one
            else
            {
                int invented_word_lenght = UnityEngine.Random.Range(0, MaxWordLenght);
                int[] invented_word_letters = new int[invented_word_lenght];
                for(int i = 0; i < invented_word_lenght; ++i)
                {
                    // 30 is hardcoded as the max possible known letter
                    invented_word_letters[i] += UnityEngine.Random.Range(0, 30); 
                }

                if (IS_DEBUG)
                {
                    input_debug = GetWordByIntArray(invented_word_letters);
                }

                x_val = NormalizeWord(invented_word_letters);
                t_val[0] = 0;
            }
        }

        public override double[] Get_t_values(int index)
        {
            throw new System.NotImplementedException();
        }

        public override double[] Get_x_values(int index)
        {
            throw new System.NotImplementedException();
        }

    }
}


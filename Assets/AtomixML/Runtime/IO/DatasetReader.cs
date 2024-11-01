using Atom.MachineLearning.Core;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Plastic.Newtonsoft.Json;
using UnityEngine;

namespace Atom.MachineLearning.IO
{
    /// <summary>
    /// Base class for reading datas from a dataset and feed training algorithm
    /// </summary>
    public static class DatasetReader
    {
        public static string[,] ReadCSV(string filepath, char separator, string[] headers = null)
        {
            //read
            var lines = File.ReadAllLines(filepath).ToList();
            //clean empty
            lines.RemoveAll(t => t == string.Empty);

            string[] text = lines.ToArray();

            if (headers == null)
            {
                var columns = text.First().Split(separator, StringSplitOptions.RemoveEmptyEntries);
                string[,] datas = new string[text.Length, columns.Length];

                for (int i = 0; i < text.Length; ++i)
                {
                    var row = text[i].Split(separator, StringSplitOptions.RemoveEmptyEntries);

                    for (int j = 0; j < row.Length; ++j)
                        datas[i, j] = row[j];
                }

                return datas;
            }

            throw new NotImplementedException();
        }

        public static void Split(string[,] datas, int splitIndex, out string[,] right, out string[,] left)
        {
            right = new string[datas.GetLength(0), splitIndex + 1];
            left = new string[datas.GetLength(0), datas.GetLength(1) - splitIndex - 1];
            int left_start_index = splitIndex + 1;
            int total_lenght = datas.GetLength(1);

            for (int i = 0; i < datas.GetLength(0); ++i)
            {
                for (int j = 0; j < left_start_index; ++j)
                    right[i, j] = datas[i, j];

                for (int k = left_start_index; k < total_lenght; ++k)
                    left[i, k - left_start_index] = datas[i, k];
            }
        }

        public static void SplitLastColumn(string[,] datas, out string[,] right, out string[] left)
        {
            int total_lenght = datas.GetLength(1);
            int splitIndex = total_lenght - 2;

            right = new string[datas.GetLength(0), total_lenght - 1];
            left = new string[datas.GetLength(0)];

            int left_start_index = splitIndex + 1;

            for (int i = 0; i < datas.GetLength(0); ++i)
            {
                for (int j = 0; j < left_start_index; ++j)
                    right[i, j] = datas[i, j];

                left[i] = datas[i, total_lenght -1];
            }
        }

        public static T ReadJSON<T>(string filepath)
        {
            string json = File.ReadAllText(filepath);
            var clas = JsonConvert.DeserializeObject<T>(json);
            //////////
            ////////
            return clas;
        }

        public static List<Texture2D> ReadTextures(string folderpath)
        {
            return Resources.LoadAll<Texture2D>(folderpath).ToList();
        }
    }
}

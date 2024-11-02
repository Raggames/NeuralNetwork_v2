using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

namespace Atom.MachineLearning.IO
{
    /// <summary>
    /// Base class for reading datas from a dataset and feed training algorithm
    /// </summary>
    public static class DatasetReader
    {
        /// <summary>
        /// Reads a csv file at path, and split it with separator char
        /// </summary>
        /// <param name="filepath"></param>
        /// <param name="separator"></param>
        /// <param name="headersCount"> Tells if the csv has header row(s), which should be skipped </param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        public static string[,] ReadCSV(string filepath, char separator, int headersCount = 0)
        {
            //read
            var lines = File.ReadAllLines(filepath).ToList();
            //clean empty
            lines.RemoveAll(t => t == string.Empty);

            string[] text = lines.ToArray();

            var columns = text.First().Split(separator, StringSplitOptions.RemoveEmptyEntries);
            string[,] datas = new string[text.Length, columns.Length];

            for (int i = headersCount; i < text.Length; ++i)
            {
                var row = text[i].Split(separator, StringSplitOptions.RemoveEmptyEntries);

                for (int j = 0; j < row.Length; ++j)
                    datas[i, j] = row[j];
            }

            return datas;
        }

        /// <summary>
        /// Allow, for instance, subselection of a set into a training rows and test rows
        /// </summary>
        /// <param name="datas"></param>
        /// <param name="splitIndex"></param>
        /// <param name="right"></param>
        /// <param name="left"></param>
        public static void SplitRows(string[,] datas, int splitIndex, out string[,] right, out string[,] left)
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

        /// <summary>
        /// Allows, for instance, to split data columns from label columns
        /// </summary>
        /// <param name="datas"></param>
        /// <param name="splitIndex"></param>
        /// <param name="right"></param>
        /// <param name="left"></param>
        public static void SplitColumn(string[,] datas, int splitIndex, out string[,] right, out string[,] left)
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

        /// <summary>
        /// Allows, for instance, to split data columns from label unique column
        /// </summary>
        /// <param name="datas"></param>
        /// <param name="splitIndex"></param>
        /// <param name="right"></param>
        /// <param name="left"></param>
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

                left[i] = datas[i, total_lenght - 1];
            }
        }

        /// <summary>
        /// Returns textures from Unity's resource folder
        /// </summary>
        /// <param name="folderpath"></param>
        /// <returns></returns>
        public static List<Texture2D> ReadTextures(string folderpath)
        {
            return Resources.LoadAll<Texture2D>(folderpath).ToList();
        }
    }
}

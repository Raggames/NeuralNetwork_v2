using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

namespace Atom.MachineLearning.IO
{
    /// <summary>
    /// Base class for reading datas from a dataset and feed training algorithm
    /// </summary>
    public static class DatasetRWUtils
    {
        /// <summary>
        /// Returns textures from Unity's resource folder
        /// </summary>
        /// <param name="folderpath"></param>
        /// <returns></returns>
        public static List<Texture2D> ReadTextures(string folderpath)
        {
            return Resources.LoadAll<Texture2D>(folderpath).ToList();
        }

        /// <summary>
        /// Reads a csv file at path, and split it with separator char
        /// </summary>
        /// <param name="filepath"></param>
        /// <param name="separator"></param>
        /// <param name="startIndex"> Skipping first rows (header rows ?) </param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        public static string[,] ReadCSV(string filepath, char separator, int startIndex = 1)
        {
            //read
            var lines = File.ReadAllLines(filepath).ToList();
            //clean empty
            lines.RemoveAll(t => t == string.Empty);

            string[] text = lines.ToArray();

            var columns = text.First().Split(separator);
            string[,] datas = new string[text.Length - startIndex, columns.Length];

            for (int i = startIndex; i < text.Length; ++i)
            {
                var row = text[i].Split(separator);

                for (int j = 0; j < row.Length; ++j)
                    datas[i - startIndex, j] = row[j];
            }

            return datas;
        }

        /// <summary>
        /// Writes a CSV file to the specified filepath.
        /// </summary>
        /// <param name="filepath">Path to the CSV file to write.</param>
        /// <param name="separator">Character used to separate columns (e.g., comma, semicolon).</param>
        /// <param name="headers">Array of column headers.</param>
        /// <param name="datas">2D array of string data (rows and columns).</param>
        public static void WriteCSV(string filepath, char separator, string[] headers, string[,] datas)
        {
            if (headers == null || headers.Length == 0)
                throw new ArgumentException("Headers cannot be null or empty.", nameof(headers));

            if (datas == null || datas.GetLength(1) != headers.Length)
                throw new ArgumentException("Data columns must match the header count.", nameof(datas));

            // Use StringBuilder to construct the CSV content
            StringBuilder csvContent = new StringBuilder();

            // Write headers
            csvContent.AppendLine(string.Join(separator, headers));

            // Write data rows
            for (int row = 0; row < datas.GetLength(0); row++)
            {
                string[] rowData = new string[datas.GetLength(1)];
                for (int col = 0; col < datas.GetLength(1); col++)
                {
                    // Escape separator characters if necessary
                    string cell = datas[row, col]?.Replace(separator.ToString(), "\\" + separator) ?? "";
                    rowData[col] = cell;
                }
                csvContent.AppendLine(string.Join(separator, rowData));
            }

            // Write to file
            try
            {
                File.WriteAllText(filepath, csvContent.ToString());
            }
            catch (Exception ex)
            {
                throw new IOException($"Error writing to file {filepath}: {ex.Message}", ex);
            }
        }

        public static void Split_TrainTest_StringArray(string[,] datas, float splitRatio01, out string[,] before, out string[,] after)
        {
            int split_index = (int)Math.Round((datas.Length * splitRatio01));
            Split_TrainTest_StringArray(datas, split_index, out before, out after);
        }

        /// <summary>
        /// Allow, for instance, subselection of a set into a training rows and test rows
        /// </summary>
        /// <param name="datas"></param>
        /// <param name="splitIndex"></param>
        /// <param name="right"></param>
        /// <param name="left"></param>
        public static void Split_TrainTest_StringArray(string[,] datas, int splitIndex, out string[,] before, out string[,] after)
        {
            //ShuffleRows(datas);

            before = new string[splitIndex, datas.GetLength(1)];
            after = new string[datas.GetLength(0) - splitIndex, datas.GetLength(1)];

            for (int i = 0; i < splitIndex; ++i)
            {
                for (int j = 0; j < datas.GetLength(1); ++j)
                {
                    before[i, j] = datas[i, j];
                }
            }

            int index = 0;
            for (int i = splitIndex; i < datas.GetLength(0); ++i)
            {
                for (int j = 0; j < datas.GetLength(1); ++j)
                {
                    after[index, j] = datas[i, j];
                }

                index++;
            }
        }

        public static void Split_TrainTest_NVector(NVector[] datas, float splitRatio01, out NVector[] train, out NVector[] test)
        {
            int split_index = (int)Math.Round((datas.Length * splitRatio01));
            Split_TrainTest_NVector(datas, split_index, out train, out test);
        }


        /// <summary>
        /// Allow, for instance, subselection of a set into a training rows and test rows
        /// </summary>
        /// <param name="datas"></param>
        /// <param name="splitIndex"></param>
        /// <param name="right"></param>
        /// <param name="left"></param>
        public static void Split_TrainTest_NVector(NVector[] datas, int splitIndex, out NVector[] train, out NVector[] test)
        {
            train = new NVector[splitIndex];
            test = new NVector[datas.GetLength(0) - splitIndex];
            int dimensions = datas[0].Length;

            //ShuffleRows(datas);

            for (int i = 0; i < splitIndex; ++i)
            {
                train[i] = new NVector(dimensions);

                for (int j = 0; j < dimensions; ++j)
                {
                    train[i][j] = datas[i][j];
                }
            }

            int index = 0;
            for (int i = splitIndex; i < datas.GetLength(0); ++i)
            {
                test[index] = new NVector(dimensions);

                for (int j = 0; j < dimensions; ++j)
                {
                    test[index][j] = datas[i][j];
                }

                index++;
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
        /// Shuffle rows of the matrix 
        /// </summary>
        /// <param name="datas"></param>
        public static void ShuffleRows(string[,] datas)
        {
            int rowCount = datas.GetLength(0); // Number of rows
            int colCount = datas.GetLength(1); // Number of columns

            for (int i = 0; i < rowCount; i++)
            {
                // Pick a random row to swap with
                int j = MLRandom.Shared.Next(i, rowCount);

                // Swap row i with row j
                for (int k = 0; k < colCount; k++)
                {
                    string temp = datas[i, k];
                    datas[i, k] = datas[j, k];
                    datas[j, k] = temp;
                }
            }
        }

        /// <summary>
        /// Shuffle rows of the matrix 
        /// </summary>
        /// <param name="datas"></param>
        public static void ShuffleRows(double[,] datas)
        {
            int rowCount = datas.GetLength(0); // Number of rows
            int colCount = datas.GetLength(1); // Number of columns

            for (int i = 0; i < rowCount; i++)
            {
                // Pick a random row to swap with
                int j = MLRandom.Shared.Next(i, rowCount);

                // Swap row i with row j
                for (int k = 0; k < colCount; k++)
                {
                    double temp = datas[i, k];
                    datas[i, k] = datas[j, k];
                    datas[j, k] = temp;
                }
            }
        }


        /// <summary>
        /// Shuffle rows of the matrix 
        /// </summary>
        /// <param name="datas"></param>
        public static void ShuffleRows(NVector[] datas)
        {
            int rowCount = datas.GetLength(0); // Number of rows

            for (int i = 0; i < rowCount; i++)
            {
                // Pick a random row to swap with
                int j = MLRandom.Shared.Next(i, rowCount);
                var temp = datas[i];
                datas[i] = datas[j];
                datas[j] = temp;
            }
        }

    }
}

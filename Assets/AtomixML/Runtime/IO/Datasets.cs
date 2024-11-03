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
    }
}

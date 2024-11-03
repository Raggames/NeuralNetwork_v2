using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.IO
{
    public static class Datasets
    {
        public static string[,] Flowers()
        {
            return DatasetReader.ReadCSV("Assets/AtomixML/Resources/Datasets/flowers/iris.data.txt", ',', 0);
        }

        public static string[,] AmericanHousing_Train()
        {
            return null;
        }

        public static string[,] AmericanHousing_Test()
        {
            return null;
        }
    }
}

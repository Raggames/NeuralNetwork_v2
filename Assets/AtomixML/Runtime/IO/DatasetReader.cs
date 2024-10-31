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
    /// Base class for reading datas from a dataset and feed training algorithm
    /// </summary>
    public static class DatasetReader  
    {
        public static List<T> ReadCSV<T>(string filepath)
        {
            throw new Exception();
        }

        public static List<T> ReadJSON<T>(string filepath)
        {
            throw new Exception();
        }   
        
        public static List<Texture2D> ReadTextures(string folderpath)
        {
            return Resources.LoadAll<Texture2D>(folderpath).ToList();
        }
    }
}

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace NeuralNetwork
{
    public class WordReader
    {
        private string file_path;

        public WordReader(string filer_Path)
        {
            file_path = filer_Path;
        }

        public string[] Read()
        {
            string[] files = Directory.GetFiles(Application.dataPath + "/StreamingAssets/Dictionnaries");
            string file = files.ToList().Find(t => t.Contains("fr.txt"));

            if (File.Exists(file))
            {
                //string LoadJson = File.ReadAllText(Application.dataPath + "/StreamingAssets/Dictionnaries" + dictionnaryFileName); 
                return File.ReadAllLines(file);
            }

            return null;
        }
    }
}

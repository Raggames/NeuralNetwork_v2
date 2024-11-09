using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;


namespace Atom.MachineLearning.NeuralNetwork
{
    public static class NetworkDataSerializer
    {
        public static void Save(NetworkData gameData, string fileName)
        {
            string saveJson = JsonUtility.ToJson(gameData);
            string path = Application.dataPath + "/StreamingAssets/" + fileName;
            File.WriteAllText(path, saveJson);
            Debug.Log("NetData Saved => " + path);
        }

        public static NetworkData Load(NetworkData netData, string fileName)
        {
            string[] files = Directory.GetFiles(Application.dataPath + "/StreamingAssets/");
            string file = files.ToList().Find(t => t.Contains(fileName));

            NetworkData loadedData = new NetworkData();
            if (File.Exists(file))
            {
                string LoadJson = File.ReadAllText(file);
                loadedData = JsonUtility.FromJson<NetworkData>(LoadJson);
                Debug.Log("Loaded :" + fileName);
            }
            else
            {
                Debug.LogError("File not found");
            }
            return loadedData;
        }
    }
}

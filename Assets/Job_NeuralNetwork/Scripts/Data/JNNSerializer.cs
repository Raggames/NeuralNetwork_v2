using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using static Assets.Job_NeuralNetwork.Scripts.NeuralNetwork;

namespace Assets.Job_NeuralNetwork.Scripts
{
    public static class JNNSerializer
    {
        public static void Save(NetworkData gameData, string fileName)
        {
            string saveJson = JsonUtility.ToJson(gameData);
            File.WriteAllText(Application.dataPath + "/StreamingAssets/" + fileName, saveJson);
            Debug.Log("NetData Saved");
        }

        public static NetworkData Load(NetworkData netData, string fileName)
        {
            NetworkData loadedData = new NetworkData();
            if (File.Exists(Application.dataPath + "/StreamingAssets/" + fileName))
            {
                string LoadJson = File.ReadAllText(Application.dataPath + "/StreamingAssets/" + fileName);
                loadedData = JsonUtility.FromJson<NetworkData>(LoadJson);
                Debug.Log("Loaded :" + fileName);
            }
            return loadedData;
        }
    }
}

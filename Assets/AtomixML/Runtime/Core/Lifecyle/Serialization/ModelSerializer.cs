using Newtonsoft.Json;
using System;
using System.IO;
using System.Linq;
using UnityEngine;

namespace Atom.MachineLearning.Core
{
    /// <summary>
    /// A manager that handle serialization and creation/deserialization of models
    /// </summary>
    public static class ModelSerializer
    {
        private static string _relativePath = "/AtomixML/Resources/Models/";
        private static string _rootPath => Application.dataPath + _relativePath;

        /// <summary>
        /// Return an instance of the model by its name and (optional) its version.
        /// Versions are automatically incremented by trainers when autosaving
        /// If version isn't specified, the service will return the latest avalaible version in the directory
        /// </summary>
        /// <typeparam name="TModelClass"></typeparam>
        /// <param name="modelName"></param>
        /// <param name="version"></param>
        public static TModelClass LoadModel<TModelClass>(string modelName, string version = null) where TModelClass : IMLModelCore
        {
            string model_impl_path = GetOrCreateDirectories<TModelClass>(modelName);

            if (version != null)
            {
                string modelFilePath = Path.Combine(model_impl_path, $"_{version}.json");
                if (File.Exists(modelFilePath))
                {
                    string json = File.ReadAllText(modelFilePath);
                    return JsonConvert.DeserializeObject<TModelClass>(json);
                }

                throw new Exception($"No model {modelName} found with version {version}");
            }

            // Find the latest version
            var versions = Directory.GetFiles(model_impl_path);
            var latestVersion = versions.OrderByDescending(v => v).FirstOrDefault();
            if (latestVersion != null)
            {
                string modelFilePath = latestVersion;
                string json = File.ReadAllText(modelFilePath);
                return JsonConvert.DeserializeObject<TModelClass>(json);
            }

            throw new Exception($"No model {modelName} found in directory");
        }


        public static void SaveModel<TModelClass>(TModelClass model, string version = null) where TModelClass : IMLModelCore
        {
            string model_impl_path = GetOrCreateDirectories<TModelClass>(model.ModelName);
            string json = string.Empty;

            if (version != null)
            {
                model.ModelVersion = version;
            }
            else version = model.ModelVersion;

            var versions = Directory.GetFiles(model_impl_path, ".json");
            var latestVersion = versions?.OrderByDescending(v => v).FirstOrDefault();
            if (latestVersion != null)
            {
                string version_string = latestVersion.Split('_').Last().Replace(".json", "");
                version = IncrementVersionString(version_string);
                model.ModelVersion = version;
            }
                        
            if (version == null)
                version = "1.0.0";

            string modelFilePath =  $"{model_impl_path}/{model.ModelName}_{version}.json";
                        
            json = JsonConvert.SerializeObject(model);
            File.WriteAllText(modelFilePath, json);
        }

        private static string IncrementVersionString(string versionString)
        {
            var split = versionString.Split('.');
            var last_number = int.Parse(split.Last());
            split[split.Length - 1] = (last_number++).ToString();
            var new_string = string.Join('.', split);

            return new_string;
        }

        private static string GetOrCreateDirectories<TModelClass>(string modelName) where TModelClass : IMLModelCore
        {
            var model_type_directory_path = _rootPath + typeof(TModelClass).Name;
            if (!Directory.Exists(model_type_directory_path))
                Directory.CreateDirectory(model_type_directory_path);

            var model_impl_path = model_type_directory_path + "/" + modelName;
            if (!Directory.Exists(model_impl_path))
                Directory.CreateDirectory(model_impl_path);

            return model_impl_path;
        }
    }
}

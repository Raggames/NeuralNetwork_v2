using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using Newtonsoft.Json;
using UnityEngine;
using UnityEngine.Networking;


namespace Atom.MachineLearning.IO 
{ 
    /// <summary>
    /// Lit un fichier contenu dans les streamingAssets
    /// </summary>
    public class CSVReaderService
    {
        private char _separator = ',';

        private Encoding _encoding = Encoding.UTF8;
        public Encoding Encoding
        {
            get { return _encoding; }
            set
            {
                _encoding = value;
            }
        }

        /// <summary>
        /// Returns a class containing a List<> Datas of the model of the data retrieved from the csv file as a collection of objects
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="filename"></param>
        /// <returns></returns>
        public T GetData<T>(string filename)
        {
            return JsonConvert.DeserializeObject<T>(GetCsvFileJson(filename));
        }

        private string GetCsvFileJson(string fileName)
        {
            // La lecture du fichier est asynchrone, car on peut avoir des compilations WEB GL où les données sont distantes
            (string[], List<string[]>) datas = ReadCsvFile(fileName);
            var headers = datas.Item1;
            var values  = datas.Item2;

            // On transforme l'output CSV en dictionnaire json associant Nom de colonne et Valeur pour chaque ligne afin de pouvoir le désérialiser sous forme d'un objet C#
            string json = CreateJson(headers, values);
            var json_string_formatted = "{\"Datas\":" + json + "}";

            return json_string_formatted;
        }

        /// <summary>
        /// Reads a CSV file and returns a List of object reprensenting each LINE of the file.
        /// </summary>
        /// <param name="path">The complete path of the file </param>
        /// <returns></returns>
        private (string[], List<string[]>) ReadCsvFile(string fileName) // out string[] headers)
        {
            //Debug.Log("Chemin du fichier " + fileName);

            //string fileData = System.IO.File.ReadAllText(path);

            string fileData = System.IO.File.ReadAllText(fileName, Encoding);

            string[] lines = fileData.Split("\n");
            var results = new List<string[]>();

            // Les headers sont la liste des noms de colonnes extraites de la première ligne du fichier CSV
            var headers = lines[0].Split(_separator, StringSplitOptions.RemoveEmptyEntries);

            //Debug.Log("HEADERS : " + string.Join('/', headers));

            // Pour chaque ligne du CSV, on ajoute un tableau de string à la liste
            for (int i = 1; i < lines.Length - 1; i++)       // i = 2 car les deux premières lignes sont réservées aux titres des colonnes
            {
                string[] _splittedLine = lines[i].Split(_separator, headers.Length);

                bool add = false;
                for (int j = 0; j < _splittedLine.Length; ++j)
                {
                    if (_splittedLine[j] != string.Empty)
                    {
                        add = true;
                        break;
                    }
                }

                if (add)
                {
                    results.Add(_splittedLine);
                }
                else
                {
                    Debug.Log("Skipping line {i} as it is empty. ");
                }
            }

            // On retourne les résultats de la lecture par la callback
            return new (headers, results);
        }

        /// <summary>
        /// Crée un dictionnaire en associant header-value pour toutes les lignes du fichiers
        /// Cela permet de sérialiser sous une forme 'objet', qui pourra être déserialisée en une liste d'objets c# 
        /// </summary>
        /// <param name="headers"></param>
        /// <param name="fileStrings"></param>
        /// <returns></returns>
        private string CreateJson(string[] headers, List<string[]> fileStrings)
        {
            List<Dictionary<string, string>> dictionnaryList = AppendAsDictionnary(headers, fileStrings);

            string json = JsonConvert.SerializeObject(dictionnaryList);
            return json;
        }

        private List<Dictionary<string, string>> AppendAsDictionnary(string[] headers, List<string[]> fileStrings)
        {
            var dictionnaryList = new List<Dictionary<string, string>>();

            for (int i = 0; i < fileStrings.Count; ++i)
            {
                dictionnaryList.Add(new Dictionary<string, string>());
                for (int j = 0; j < fileStrings[i].Length; ++j)
                {
                    dictionnaryList[i].Add(headers[j], fileStrings[i][j]);
                }
            }

            return dictionnaryList;
        }
    }
}

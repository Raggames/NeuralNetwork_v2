using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Transformers;
using Atom.MachineLearning.IO;
using Atom.MachineLearning.NeuralNetwork;
using Atom.MachineLearning.NeuralNetwork.V2;
using Atom.MachineLearning.Unsupervised.AutoEncoder;
using Sirenix.OdinInspector;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.RecommenderSystem
{
    public class AE_Recommender : MonoBehaviour
    {
        [SerializeField] private AutoEncoderTrainer _trainer;
        [SerializeField] private float _visualizationUpdateTimer = .05f;
        [SerializeField] private string _datasetsFolderPath = "Assets/AtomixML/MiniProjects/RecommenderSystem/Resources";
        [SerializeField] private string _userToItemProfilesCsvPath = "Assets/AtomixML/MiniProjects/RecommenderSystem/Resources/uiProf.csv";

        [SerializeField, ValueDropdown(nameof(getAvalaibleDatasets))] private string _datasetPath;

        private IEnumerable getAvalaibleDatasets()
        {
            var files = Directory.GetFiles(_datasetsFolderPath).Where(t => t.Contains("dataset"));
            return files;
        }

        private NVector[] _ratingsDataset;

        [Button]
        private async void Continue()
        {
            await _trainer.Fit(_ratingsDataset);
        }

        [Button]
        private void Cancel()
        {
            StopAllCoroutines();
            _trainer.Cancel();
        }

        [Button]
        private async void Fit()
        {
            var datas = DatasetRWUtils.ReadCSV(_datasetPath, ';', 1);

            var check_data_name = _datasetPath.Replace(".csv", "");
            check_data_name += "_check.csv";
            var user_type_datas = DatasetRWUtils.ReadCSV(check_data_name, ';', 1);
            var uiProfilesCsv = DatasetRWUtils.ReadCSV(_userToItemProfilesCsvPath, ',', 1);
            var profiles = new FeaturesParser().Transform(new FeaturesSelector(Enumerable.Range(1, 26).ToArray()).Remap("0", "0,5").Transform(uiProfilesCsv));

            _ratingsDataset = new FeaturesParser().Transform(datas.ToNStringVectorArray());
            var normalizer = new TrMinMaxNormalizer();
            _ratingsDataset = normalizer.Transform(_ratingsDataset);

            int features = _ratingsDataset[0].Length;

            var encoder = new NeuralNetworkModel();
            encoder.AddDenseLayer(features, 25, ActivationFunctions.Tanh, (x) => x);
            encoder.AddDenseLayer(6, ActivationFunctions.Sigmoid, (x) => x);
            encoder.SeedWeigths();
            var decoder = new NeuralNetworkModel();
            decoder.AddDenseLayer(6, 25, ActivationFunctions.Sigmoid, (x) => x);
            decoder.AddOutputLayer(features, ActivationFunctions.Tanh, (x) => x);
            decoder.SeedWeigths();
            _trainer.trainedModel = new AutoEncoderModel(encoder, decoder);

            _trainer.trainedModel.ModelName = "auto-encoder-basic-ae-recommender";
            _trainer.SetLossFunction(LossFunctions.MaskedMeanSquarredError);

            await _trainer.Fit(_ratingsDataset);

            Debug.Log("End fit");


        }

    }
}

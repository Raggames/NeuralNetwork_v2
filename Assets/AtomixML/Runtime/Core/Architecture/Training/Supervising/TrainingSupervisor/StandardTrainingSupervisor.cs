using Atom.MachineLearning.Core.Maths;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Core.Training
{
    /// <summary>
    /// A supervisor to manage epoch and trainset iterating, model saving, etc..
    /// </summary>
    public class StandardTrainingSupervisor : ITrainingSupervisor
    {
        private IEpochIteratable _epochIteratable;
        private ITrainIteratable _trainIteratable;
        private IBatchedTrainIteratable _batchedTrainIteratable;

        private CancellationTokenSource _cancellationTokenSource;

        private bool _shuffleTrainIndex = false;

        public StandardTrainingSupervisor()
        {
        }

        public ITrainingSupervisor SetEpochIteration(IEpochIteratable target)
        {
            _epochIteratable = target;
            return this;
        }

        public ITrainingSupervisor SetTrainIteration(ITrainIteratable target)
        {
            _trainIteratable = target;
            return this;
        }

        public ITrainingSupervisor SetTrainBatchIteration(IBatchedTrainIteratable target)
        {
            _batchedTrainIteratable = target;
            return this;
        }

        public ITrainingSupervisor SetAutosave(int epoch_interval = 1)
        {

            return this;
        }

        public void Cancel()
        {
            _cancellationTokenSource?.Cancel();
        }

        public async Task RunOnlineAsync(int epochs, int trainLenght = 0, bool shuffleTrainIndex = true)
        {
            if (_cancellationTokenSource != null)
            {
                _cancellationTokenSource.Cancel();
                throw new Exception($"A process is currently executing.");
            }

            _cancellationTokenSource = new CancellationTokenSource();
            _shuffleTrainIndex = shuffleTrainIndex;

            if (_trainIteratable != null && trainLenght != 0)
            {
                await Task.Factory.StartNew(() => OnlineRunner(epochs, trainLenght, _cancellationTokenSource.Token));
            }
            else throw new Exception($"Supervisor should be initialized with iterators");
        }

        public async Task RunBatchedAsync(int epochs, int trainLength, int batchSize, bool shuffleTrainIndex = true)
        {
            if (_cancellationTokenSource != null)
            {
                _cancellationTokenSource.Cancel();
                throw new Exception($"A process is currently executing.");
            }

            _cancellationTokenSource = new CancellationTokenSource();
            _shuffleTrainIndex = shuffleTrainIndex;

            if (_batchedTrainIteratable != null)
            {
                await Task.Factory.StartNew(() => BatchRunner(epochs, trainLength, batchSize, _cancellationTokenSource.Token));
            }
            else throw new Exception($"Supervisor should be initialized with iterators");
        }

        public async Task RunEpochAsync(int epochs, bool shuffleTrainIndex = true)
        {
            if (_cancellationTokenSource != null)
            {
                _cancellationTokenSource.Cancel();
                throw new Exception($"A process is currently executing.");
            }

            _cancellationTokenSource = new CancellationTokenSource();
            _shuffleTrainIndex = shuffleTrainIndex;

            if (_epochIteratable != null)
            {
                await Task.Factory.StartNew(() => EpochRunner(epochs, _cancellationTokenSource.Token));
            }
            else throw new Exception($"Supervisor should be initialized with iterators");
        }

        private void OnlineRunner(int epochs, int trainLenght, CancellationToken cancellationToken)
        {
            if (_shuffleTrainIndex)
            {
                var indexes = new List<int>();

                for (int i = 0; i < epochs; i++)
                {
                    indexes.AddRange(Enumerable.Range(0, trainLenght));

                    _epochIteratable.OnBeforeEpoch(i);

                    while (indexes.Count > 0)
                    {
                        int index = MLRandom.Shared.Range(0, indexes.Count);
                        int next_train_shuffled_index = indexes[index];
                        _trainIteratable.OnTrainNext(next_train_shuffled_index);
                        cancellationToken.ThrowIfCancellationRequested();

                        indexes.RemoveAt(index);
                    }

                    _epochIteratable.OnAfterEpoch(i);
                    cancellationToken.ThrowIfCancellationRequested();
                }

                _cancellationTokenSource = null;
            }
            else
            {

                for (int i = 0; i < epochs; i++)
                {
                    _epochIteratable.OnBeforeEpoch(i);

                    for (int j = 0; j < trainLenght; j++)
                    {
                        _trainIteratable.OnTrainNext(j);
                        cancellationToken.ThrowIfCancellationRequested();
                    }

                    _epochIteratable.OnAfterEpoch(i);
                    cancellationToken.ThrowIfCancellationRequested();
                }

                _cancellationTokenSource = null;
            }
        }

        public void BatchRunner(int epochs, int trainLenght, int batchSize, CancellationToken cancellationToken)
        {
            try
            {
                var batchIndexes = new int[batchSize];

                if (_shuffleTrainIndex)
                {
                    var indexes = new List<int>();

                    for (int i = 0; i < epochs; i++)
                    {
                        indexes.AddRange(Enumerable.Range(0, trainLenght));

                        _batchedTrainIteratable.OnBeforeEpoch(i);

                        while (indexes.Count > 0)
                        {
                            int index = 0;

                            try
                            {

                                for (int j = 0; j < batchSize; j++)
                                {
                                    index = MLRandom.Shared.Range(0, indexes.Count);

                                    if (index < 0 || index >= indexes.Count)
                                        continue;

                                    batchIndexes[j] = indexes[index];
                                    indexes.RemoveAt(index);
                                }

                            }
                            catch (Exception e)
                            {

                            }

                            _batchedTrainIteratable.OnTrainNextBatch(batchIndexes);
                            cancellationToken.ThrowIfCancellationRequested();
                        }

                        _batchedTrainIteratable.OnAfterEpoch(i);
                        cancellationToken.ThrowIfCancellationRequested();
                    }

                    _cancellationTokenSource = null;
                }
                else
                {

                    for (int i = 0; i < epochs; i++)
                    {
                        _batchedTrainIteratable.OnBeforeEpoch(i);

                        for (int j = 0; j < trainLenght - batchSize; j += batchSize)
                        {
                            for (int k = 0; k < batchSize; k++)
                            {
                                int index = j + k;
                                batchIndexes[k] = index;
                            }

                            _batchedTrainIteratable.OnTrainNextBatch(batchIndexes);
                            cancellationToken.ThrowIfCancellationRequested();
                        }

                        _batchedTrainIteratable.OnAfterEpoch(i);
                        cancellationToken.ThrowIfCancellationRequested();
                    }

                    _cancellationTokenSource = null;
                }

            }
            catch (Exception ex)
            {
                Debug.Log(ex);
            }

        }

        private void EpochRunner(int epochs, CancellationToken cancellationToken)
        {
            for (int i = 0; i < epochs; i++)
            {
                _epochIteratable.OnBeforeEpoch(i);
                cancellationToken.ThrowIfCancellationRequested();
                _epochIteratable.OnAfterEpoch(i);
                cancellationToken.ThrowIfCancellationRequested();
            }

            _cancellationTokenSource = null;
        }
    }
}

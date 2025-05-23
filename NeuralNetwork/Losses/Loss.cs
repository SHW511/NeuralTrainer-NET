﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Losses
{
    [Serializable]
    public abstract class Loss
    {
        public abstract float Calculate(float[,] predicted, float[,] actual);
        public abstract float Calculate4D(float[,,,] predicted, float[,] actual);

    }
}

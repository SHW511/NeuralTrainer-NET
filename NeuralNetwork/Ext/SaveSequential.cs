using System;
using System.IO;
using System.Reflection;
using System.Text.Json;
using System.Xml.Serialization;
using NeuralNetwork.Layers;
using NeuralNetwork.Losses;
using NeuralNetwork.Optimizers;

namespace NeuralNetwork.Ext
{
    public static class SaveSequential
    {
        public static void Save(this Sequential model, string name, string path = "")
        {
            XmlSerializer serializer = new XmlSerializer(typeof(Sequential), [typeof(MeanSquaredError), typeof(SGD), typeof(Dense)]);

            if (string.IsNullOrEmpty(path))
            {
                path = Path.Combine(Directory.GetCurrentDirectory(), $"{name}.xml");
            }
            else
            {
                path = Path.Combine(path, $"{name}.xml");
            }

            using (TextWriter writer = new StreamWriter(path))
            {
                serializer.Serialize(writer, model);
            }
        }
    }
}

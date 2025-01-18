using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Losses;
using NeuralNetwork.Optimizers;
using System.Xml.Serialization;

namespace NeuralNetwork.Ext
{
    public static class LoadSequential
    {
        public static void Load(this Sequential model, string filename, string path = "")
        {
            if (string.IsNullOrEmpty(filename))
            {
                throw new ArgumentNullException("Path cannot be null or empty", nameof(filename));
            }

            if (string.IsNullOrWhiteSpace(path))
            {
                path = Path.Combine(Directory.GetCurrentDirectory(), filename);
            }

            if (!File.Exists(path))
            {
                throw new FileNotFoundException("File not found", path);
            }

            XmlSerializer serializer = new XmlSerializer(typeof(Sequential), [typeof(Layer), typeof(MeanSquaredError), typeof(SGD)]);
            using (TextReader reader = new StreamReader(path))
            {
                model = (Sequential)serializer.Deserialize(reader);
            }
        }
    }
}

from neat_classes.network import Network
from neat_classes.neurons.input_neuron import Input_neuron
from neat_classes.neurons.output_neuron import Output_neuron
from neat_classes.HistoricalMarker import HistoricalMarker
import math

hist_marker = HistoricalMarker(3)

def test_network_traversal():
    net = Network(2,1)
    in_1 = Input_neuron(0)
    in_2 = Input_neuron(1)
    out = Output_neuron(2)
    net.input_neurons.append(in_1)
    net.input_neurons.append(in_2)
    net.output_neurons.append(out)

    net.add_new_connection(in_1, out, hist_marker)
    net.add_new_connection(in_2, out, hist_marker)

    # test simple initial network
    out.bias_weight = 10

    net.connections[0].weight = 0.3
    net.connections[1].weight = 0.5

    out_input_1 = 3 * 0.3
    out_input_2 = 5 * 0.5

    n3_weighted_input = out_input_1 + out_input_2 + 10
    out = 1 / (1 + math.pow(math.e, -n3_weighted_input))

    print(out)
    outputs = net.compute_inputs(3,5)
    print(outputs[0])
    assert out == outputs[0]



# Import modules
import itertools
import logging
from rig import machine

# Import classes
from conv_neuron_layer import ConvNeuronLayer
from rig.bitfield import BitField
from rig.machine_control.consts import AppState, signal_types, AppSignal, MessageType
from rig.machine_control.machine_controller import MachineController
from rig.netlist import Net

# Import functions
from rig.place_and_route import place_and_route_wrapper
from six import itervalues

logger = logging.getLogger("convolver")

# ----------------------------------------------------------------------------
# ConvNet
# ----------------------------------------------------------------------------
class ConvNet(object):
    def __init__(self, neuron_threshold, neuron_decay, test_data,
                 timer_period_us=1000, sim_ticks=1000):
        # Cache network parameters
        self._neuron_threshold = neuron_threshold
        self._neuron_decay = neuron_decay
        self._test_data = test_data
        self._timer_period_us = timer_period_us
        self._sim_ticks = sim_ticks

        self._vert_index = 0

        # Create data structures
        self._layers = []
        self._vertex_applications = {}
        self._vertex_resources = {}

        # Create a 32-bit keyspace
        self._keyspace = BitField(32)
        self._keyspace.add_field("vert_index", tags="routing")
        self._keyspace.add_field("z")
        self._keyspace.add_field("y", length=8, start_at=8)
        self._keyspace.add_field("x", length=8, start_at=0)

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------
    def add_layer(self, output_width, output_height, padding, stride, weights):
        # Get index of new layer
        layer_index = len(self._layers)

        # Add layer to conv net
        self._layers.append(
            ConvNeuronLayer(start_vert_index=self._vert_index,
                            output_width=output_width,
                            output_height=output_height,
                            padding=padding, stride=stride,
                            neuron_decay=self._neuron_decay,
                            neuron_threshold=self._neuron_threshold,
                            weights=weights, parent_keyspace=self._keyspace,
                            input_data=(self._test_data if layer_index == 0
                                        else None),
                            vertex_applications=self._vertex_applications,
                            vertex_resources=self._vertex_resources,
                            timer_period_us=self._timer_period_us,
                            sim_ticks=self._sim_ticks))

        # **YUCK** update vertex index
        self._vert_index += len(self._layers[-1].vertices)

    def run(self, spinnaker_hostname, disable_software_watchdog=False):
        logger.info("Assigning keyspaces")

        # Finalise keyspace fields
        self._keyspace.assign_fields()

        logger.info("Building nets")

        # Loop through layers and their successors
        nets = []
        net_keys = {}
        for layer, next_layer in zip(self._layers[:-1], self._layers[1:]):
            # Loop through all vertices in layer
            for vertex in layer.vertices:
                # Create a key for the vertex feeding forward
                net_key = (vertex.routing_key, vertex.routing_mask)

                # Create a net connecting vertex
                # to all vertices in next layer
                net = Net(vertex, next_layer.vertices)

                # Add net to list and associate with key
                nets.append(net)
                net_keys[net] = net_key

        machine_controller = None
        try:
            # Get machine controller from connected SpiNNaker board and boot
            machine_controller = MachineController(spinnaker_hostname)
            machine_controller.boot()

            # Get system info
            system_info = machine_controller.get_system_info()
            logger.debug("Found %u chip machine", len(system_info))

            # Place-and-route
            logger.info("Placing and routing")
            placements, allocations, run_app_map, routing_tables =\
                place_and_route_wrapper(self._vertex_resources,
                                        self._vertex_applications,
                                        nets, net_keys, system_info)

            # Convert placement values to a set to get unique list of chips
            unique_chips = set(itervalues(placements))
            logger.info("Placed on %u cores (%u chips)",
                        len(placements), len(unique_chips))
            logger.debug(list(itervalues(placements)))

            # If software watchdog is disabled, write zero to each chip in
            # placement's SV struct, otherwise, write default from SV struct file
            wdog = (0 if disable_software_watchdog else
                    machine_controller.structs["sv"]["soft_wdog"].default)
            for x, y in unique_chips:
                logger.debug("Setting software watchdog to %u for chip %u, %u",
                            wdog, x, y)
                machine_controller.write_struct_field("sv", "soft_wdog",
                                                    wdog, x, y)

            logger.info("Loading layers")
            z_length = self._keyspace.get_location_and_length("z")[1]
            z_mask = (1 << z_length) - 1
            logger.debug("Z length:%u, mask:%08x", z_length, z_mask)

            for l in self._layers:
                l.load(placements, allocations, machine_controller, z_mask)

            # Load routing tables and applications
            logger.info("Loading routing tables")
            machine_controller.load_routing_tables(routing_tables)

        finally:
            if machine_controller is not None:
                logger.info("Stopping SpiNNaker application")
                machine_controller.send_signal("stop")

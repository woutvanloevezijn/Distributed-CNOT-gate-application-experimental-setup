from netqasm.sdk.classical_communication.socket import Socket
from netqasm.sdk.connection import BaseNetQASMConnection
from netqasm.sdk.epr_socket import EPRSocket
from netqasm.sdk.qubit import Qubit

from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta
import netsquid as ns
import os



class ControllerProgram(Program):
    PEER_NAME = "Target"

    def __init__(self, num_epr_rounds, input_theta, input_phi, output_theta, output_phi, bit_select):
        self._num_epr_rounds = num_epr_rounds
        self._input_theta = input_theta
        self._input_phi = input_phi
        self._output_theta = output_theta
        self._output_phi = output_phi
        self._bit_select = bit_select

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="distributed_cnot",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        
        # setup connection
        csocket = context.csockets[self.PEER_NAME]
        epr_socket = context.epr_sockets[self.PEER_NAME]
        connection = context.connection

        # run the application num_epr_rounds amount of times
        results = []
        for _ in range(self._num_epr_rounds):
            
            # make sure the NetSquid simulator uses the right RNG seed independent
            # of the process it is running in
            ns.set_random_state()

            # initialize the input control qubit of the distributed CNOT
            control_qubit = Qubit(connection)
            control_qubit.rot_Y(angle=self._input_theta)
            control_qubit.rot_Z(angle=self._input_phi)

            # generate EPR-pair
            epr_qubit = epr_socket.create_keep()[0]

            # perform a local CNOT with `epr` and measure `epr`
            control_qubit.cnot(epr_qubit)
            epr_measure = epr_qubit.measure()
        
            # let back-end execute the quantum operations above
            yield from connection.flush()

            # send the outcome to target
            csocket.send_int(int(epr_measure))

            # wait for target's measurement outcome to undo potential entanglement
            # between his EPR half and the original control qubit
            target_epr = yield from csocket.recv_int()
            if int(target_epr) == 1:
                control_qubit.Z()

            #apply unitary to convert the expected output to |0>
            control_qubit.rot_Z(angle=(self._output_phi*-1))
            control_qubit.rot_Y(angle=(self._output_theta*-1))

            control_measure = control_qubit.measure()
            
            yield from connection.flush()

            # send ack
            csocket.send_int(1)
            
            # get target value
            target_measure = yield from csocket.recv_int()

            # calculate combined measure
            if self._bit_select == 2: control_measure = int(control_measure) or int(target_measure)

            result = {"epr_meas": int(epr_measure), "error": int(control_measure)}
            results.append(result)

        # calculate the mean and standard error of the batch
        avg_epr_measure = sum([results[i]["epr_meas"] for i in range(len(results))])/len(results)
        avg_error = sum([results[i]["error"] for i in range(len(results))])/len(results)
        var_error = sum([(avg_error - results[i]["error"])**2 for i in range(len(results))]) / (len(results)-1)
        return {"avg_epr_meas": avg_epr_measure, "avg_error": avg_error, "var_error": var_error}


class TargetProgram(Program):
    PEER_NAME = "Controller"

    def __init__(self, num_epr_rounds, input_theta, input_phi, output_theta, output_phi):
        self._num_epr_rounds = num_epr_rounds
        self._input_theta = input_theta
        self._input_phi = input_phi
        self._output_theta = output_theta
        self._output_phi = output_phi

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="distributed_cnot",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):

        # setup connection
        csocket: Socket = context.csockets[self.PEER_NAME]
        epr_socket: EPRSocket = context.epr_sockets[self.PEER_NAME]
        connection: BaseNetQASMConnection = context.connection

        # run the application num_epr_rounds amount of times
        results = []
        for _ in range(self._num_epr_rounds):

            # make sure the NetSquid simulator uses the right RNG seed independent
            # of the process it is running in
            ns.set_random_state()

            # generate EPR-pair
            epr_qubit = epr_socket.recv_keep()[0]
            
            # initialize input target qubit of the distributed CNOT
            target_qubit = Qubit(connection)
            target_qubit.rot_Y(angle=self._input_theta)
            target_qubit.rot_Z(angle=self._input_phi)

            # let back-end execute the quantum operations above
            yield from connection.flush()

            # wait for Controller's measurement outcome
            c_epr_measure = yield from csocket.recv_int()

            # if outcome = 1, apply an X gate on the local EPR half
            if int(c_epr_measure) == 1:
                epr_qubit.X()

            # at this point, `epr` is correlated with the control qubit on Controller's side.
            # (If Controller's control was in a superposition, `epr` is now entangled with it.)
            # use `epr` as the control of a local CNOT on the target qubit.
            epr_qubit.cnot(target_qubit)

            # undo any potential entanglement between `epr` and Controller's control qubit
            epr_qubit.H()
            t_epr_measure = epr_qubit.measure()

            #apply unitary to convert the expected output state to |0>
            target_qubit.rot_Z(angle=(self._output_phi*-1))
            target_qubit.rot_Y(angle=(self._output_theta*-1))


            target_measure = target_qubit.measure()

            yield from connection.flush()

            # Controller will do a controlled-Z based on the outcome to undo the entanglement
            csocket.send_int(int(t_epr_measure))
            
            # get ACK
            yield from csocket.recv_int()

            # send target measure to controller for combined measurement
            csocket.send_int(int(target_measure))

            result = {"epr_meas": int(t_epr_measure), "error": int(target_measure)}
            results.append(result)
        
        # calculate the mean and standard error of the batch
        avg_epr_measure = sum([results[i]["epr_meas"] for i in range(len(results))])/len(results)
        avg_error = sum([results[i]["error"] for i in range(len(results))])/len(results)
        var_error = sum([(avg_error - results[i]["error"])**2 for i in range(len(results))]) / (len(results)-1)
        return {"avg_epr_meas": avg_epr_measure, "avg_error": avg_error, "var_error": var_error}
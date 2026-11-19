

import time
import os
if os.name != "nt":
    print("This test is only for Windows, skipping TDT test ")
else:
    import tdt
    proj = tdt.DSPProject(interface="USB")  # type: ignore
    pa5_1 = tdt.util.connect_pa5(interface="USB", device_id=1)  # type: ignore
    pa5_2 = tdt.util.connect_pa5(interface="USB", device_id=1)  # type: ignore
    circuit = proj.load_circuit("tdt_circuits/abrs_v2.rcx", "RP2")
    circuit.start(0.25)
    proj.trigger('B', 'high')
    time.sleep(1)
    proj.trigger('B', 'low')

    pa5_1.SetAtten(40)
    time.sleep(1)
    pa5_1.SetAtten(120)


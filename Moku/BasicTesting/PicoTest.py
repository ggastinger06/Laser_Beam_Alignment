import serial
import time

import sys
sys.path.append(r'c:\Users\grant\Downloads\Summer Internship\Neural Network\MultiAdjust')
import picoMotor as pico 

# ALL COMMANDS ARE FOUND IN PAGE 57 of https://experimentationlab.berkeley.edu/sites/default/files/JoystickManual.pdf



ver = pico.send('ver')
if not ver:
    print("No response from driver")

# '>' means it ran
else:

    # Setup
    pico.send('ini', timeout=10) # After unplugging drivers this reinitializes them
    pico.send('jof')             # Joystick off

    pico.send('def')             # Default Parameters (v = 2 kHz, a = 32,000 steps/s^2)

    pico.send('typ a1 0=0')      # 0 = standard picomotor
    pico.send('typ a1 1=0')
    pico.send('typ a2 0=0')
    pico.send('typ a2 1=0')

    pico.send('vel a1 0=100')    # Velocity of Motor A = 250 kHz (Defined as "fine" in the manual")
    pico.send('vel a1 1=100')    # Velocity of Motor B
    pico.send('vel a2 0=100')    # Velocity of Motor A = 250 kHz (Defined as "fine" in the manual")
    pico.send('vel a2 1=100') 

    pico.send('mon')             # Enables all drivers

    # -------------Tests------------------

     # Example 1, page 99 (Turns motor a1 A clockwise)
    """
    pico.send('jof')
    pico.send('chl a1=0')
    pico.send('typ a1 0=0')
    pico.send('mpv a1 0=0')
    pico.send('vel a1 0=500')
    pico.send('acc a1 0=5000')
    pico.send('mon')
    pico.send('pos')
    pico.send('for a1')
    pico.send('go')
    pico.send('sto')
    pico.send('pos')
    pico.send('jon')
    """

    # Example 2, page 100 (To drive a motor on a1 A and a2 B simultaneously)
    """
    pico.send('jof')
    pico.send('def')
    pico.send('chl a1=0')
    pico.send('chl a2=1')
    pico.send('typ a1 0=0')
    pico.send('typ a2 1=0')
    pico.send('vel a2 0=2000')
    pico.send('vel a2 1=2000')
    pico.send('mon')
    pico.send('pos')
    pico.send('rel a1=-5000')
    pico.send('rel a2=10000')
    pico.send('go')
    time.sleep(10000/1900)
    pico.send('pos')
    pico.send('jon')
    """

    # Example 3, page 100 (Set the max and min velocities for the motors and save them)
    """
    pico.send('mpv a1 0=0')
    pico.send('mpv a1 1=0')
    pico.send('mpv a2 0=0')
    pico.send('mpv a2 1=0')

    pico.send('vel a1 0=10')
    pico.send('vel a1 1=10')
    pico.send('vel a2 0=10')
    pico.send('vel a2 1=10')

    pico.send('sav')
    """

    # Part of the test for the raster scan
    """
    length = 10000
    pos = 0
    pico.send('chl a1=0')
    pico.send(f'rel a1={-length}')
    pico.send('go')

    while (pos < length):
        pos = -pico.get_pos()
        print(pos)

    pico.send('vel a1 0=1000')
    pico.send(f'rel a1 ={(length)*1.06}') # 1.06 steps left for every step to the right
    pico.send('go')
    time.sleep((length*1.06)/950) # Divide steps by a little less than velocity for correct delay
    pico.send('sto')
    pos = -pico.get_pos()
    print(pos)
    """

    # Run an x number of steps and see how much it really runs
    """
    pico.send('chl a2=0')
    pico.send('rel a2 =-5000')
    pico.send('go a2')
    time.sleep(5000/(1*100))
    pico.send('sto')
    print(pico.get_pos())
    """

    # Right then left
    """
    pico.send('chl a1=1')
    pico.send('for a1')
    pico.send('go a1')
    time.sleep(800)
    pico.send('hal')
    pico.send('pos')

    time.sleep(5)

    pico.send('for a1')
    pico.send('go a1')
    time.sleep(800)
    pico.send('hal')
    pico.send('pos')
    """

    # Run a motor for x seconds
    """
    pico.send('chl a1=0')        # 0 = Motor A --> change this to use other motor
    pico.send('for a1')
    pico.send('go a1')

    for i in range(180, 0, -1): 
        print(f" {i} seconds left", end = "\r")
        time.sleep(1)

    pico.send('sto a1')
    """
    
    # Spiral scan test
    """
    global axis, direction, pwr, peak, peak_count

    driver = 2

    # (axis, direction, scale) - scale equalizes physical distance per direction
    # from measured 10-box strafe times: x+ 376, x- 396, y+ 271, y- 304
    if driver == 2:
        legs = [(0, 1, 1.39),   # +X 1.39   (Put offsets in the third spot)
                (1, 1, 1),      # +Y 1.00
                (0, -1, 1.46),  # -X 1.46
                (1, -1, 1.12)]  # -Y 1.12
    elif driver == 1:
         legs = [(0, 1, 1.84),  # +X 1.84   (Put offsets in the third spot)
                (1, 1, 1),      # +Y 1.00
                (0, -1, 1.53),  # -X 1.53
                (1, -1, 1.01)]  # -Y 1.01
    base_step = 100     # step size on the first ring (keep <= beam width so it isn't skipped)
    growth = 100        # step grows this much each leg, so the spiral expands faster
    seg_len = 1         # leg lengths grow 1,1,2,2,3,3,...
    leg = 0
    velocity = 1000
    pico.send(f'vel a{driver} 0={velocity}')
    pico.send(f'vel a{driver} 1={velocity}')
    peak = 0
    peak_count = 0

    while True:

        move_axis, move_dir, scale = legs[leg % 4]
        search_step = int((base_step + growth * leg) * scale)   # scaled so each direction covers equal distance

        for _ in range(seg_len):

            pico.move(move_axis, move_dir, search_step, velocity, driver)
            print(pico.get_pos())

        leg += 1
        if leg % 2 == 0:               # Grow the spiral every two legs
            seg_len += 1
        print('No signal - Scanning')
    """    

pico.cleanup
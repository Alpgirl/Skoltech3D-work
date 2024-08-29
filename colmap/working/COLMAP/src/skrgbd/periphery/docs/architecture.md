# Architecture

## Components

* _Softbox_ is an electrical unit controlled by a relay.
* _Strip_ is an electrical unit consisting of LEDs.
* _Range_ is a controllable segment of LED units on the strip.
* _Fold_ is a physical unit consisting of pillar and crossbar ranges covered by the strip folder.
* _Scaffold_ is a physical construction of pillars and crossbars as edges that make up faces.

!!! Note
    Adjacent faces share the same pillar.

## Connections

### Controller

```mermaid
graph TD
A(Arduino) ---|Pin 2| FS[Front Strip]
A ---|Pin 3| LS[Left Strip]
A ---|Pin 4| BS[Back Strip]
A ---|Pin 5| RS[Right Strip]

LSB[Left Softbox] ---|Pin 8| A
TSB[Top Softbox] ---|Pin 9| A
RSB[Right Softbox] ---|Pin 10| A
```

All pins are digital, but to control a relay one need PWM (pulse-width modulating) pin.

### Softbox

Actually, softbox is controlled by relay connected as follows

```mermaid
graph LR
PC(PC)---|USB|A(Arduino)
A----|"+5V ⟷ VCC"|R[Relay]
A----|GND|R
A----|"PWM Pin ⟷ IN"|R
```

### LED Strip

On the left end of the LED strip there are 4 inputs: `VCC`, `GND` x2, and `DI`.
Its are connected as follows

```mermaid
graph LR
PC(PC)---|USB|A(Arduino)
A----|"Digital Pin ⟷ DI"|L[LED Strip]
A----|GND|L
L----|"VCC ⟷ +12V"|P(Power Supply)
L----|GND|P
```

## Floor Plan

```mermaid
flowchart TB
subgraph Room
subgraph Scaffold
  subgraph FF[Right Fold]
    pf[pillar]
    cf[crossbar]
  end
  subgraph BF[Left Fold]
    cb[crossbar]
    pb[pillar]
  end
  subgraph RF[Front Fold]
    cr[crossbar] --- pr[pillar]
  end
  subgraph LF[Back Fold]
    pl[pillar] --- cl[crossbar]
  end
end
PC
end

LF---FF
BF---RF
LF---BF
FF---RF
```

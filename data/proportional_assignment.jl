communication_idxs=zeros(Int,2087)

APOE_total=0.3726477784114536+0.34476037624045874+0.00825145846415559+0.00627629337938669
InfDC_total=0.27179297911064065+0.2600626376708494+0.04049074209375672+0.00757843029002668
Fib_total=0.3910108498014399+0.3713174248010701+0.0146991629725874
DC_toal=0.42480376758661736+0.3785673956909513+0.04433442027072803+0.00700584514545709

APOE1=0.00627629337938669/APOE_total
APOE2=0.34476037624045874/APOE_total
APOE3=0.00825145846415559/APOE_total
APOE4=0.3726477784114536/APOE_total
InfDC1=0.00757843029002668/InfDC_total
InfDC2=0.27179297911064065/InfDC_total
InfDC3=0.04049074209375672/InfDC_total
InfDC4=0.2600626376708494/InfDC_total
Fib1=0
Fib2=0.3713174248010701/Fib_total
Fib3=0.0146991629725874/Fib_total
Fib4=0.3910108498014399/Fib_total
DC1=0.00700584514545709/DC_toal
DC2=0.3785673956909513/DC_toal
DC3=0.04433442027072803/DC_toal
DC4=0.42480376758661736/DC_toal

APOE = 1:1228
InfDC = 1229:1309
Fib = 1310:1793
DC = 1794:2087

for i in APOE
    if i < APOE1 * 1228
        communication_idxs[i] = Int(rand(APOE))
    elseif APOE1 * 1228 <= i < APOE1 * 1228+APOE2 * 1228
        communication_idxs[i] = Int(rand(InfDC))
    elseif APOE1 * 1228+APOE2 * 1228 <= i < APOE1 * 1228+APOE2 * 1228+APOE3 * 1228
        communication_idxs[i] = Int(rand(Fib))
    else
        communication_idxs[i] = Int(rand(DC))
    end
end

for i in InfDC
    if i < 1228+ InfDC1 * 80
        communication_idxs[i] = Int(rand(APOE))
    elseif 1228+InfDC1 * 80 <= i < 1228+InfDC1 * 80+InfDC2 * 80
        communication_idxs[i] = Int(rand(InfDC))
    elseif 1228+InfDC1 * 80+InfDC2 * 80 <= i < 1228+InfDC1 * 80+InfDC2 * 80+InfDC3 * 80
        communication_idxs[i] = Int(rand(Fib))
    else
        communication_idxs[i] = Int(rand(DC))
    end
end

for i in Fib
    if i < 1309+ Fib1 * 484
        communication_idxs[i] = Int(rand(APOE))
    elseif 1309+Fib1 * 484 <= i < 1309+Fib1 * 484+Fib2 * 484
        communication_idxs[i] = Int(rand(InfDC))
    elseif 1309+Fib1 * 484+Fib2 * 484 <= i < 1309+Fib1 * 484+Fib2 * 484+Fib3 * 484
        communication_idxs[i] = Int(rand(Fib))
    else
        communication_idxs[i] = Int(rand(DC))
    end
end

for i in DC
    if i < 1793+ DC1 * 294
        communication_idxs[i] = Int(rand(APOE))
    elseif 1793+DC1 * 294 <= i < 1793+DC1 * 294+DC2 * 294
        communication_idxs[i] = Int(rand(InfDC))
    elseif 1793+DC1 * 294+DC2 * 294 <= i < 1793+DC1 * 294+DC2 * 294+DC3 * 294
        communication_idxs[i] = Int(rand(Fib))
    else
        communication_idxs[i] = Int(rand(DC))
    end
end

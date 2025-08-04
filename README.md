# Generative AI for structural design of new porous materials

## 📁 [LDM-MG](https://github.com/RigijAp/LDM_MG/tree/main/LDM-MG)
Implementation of the Latent Diffusion Model for generative desing realized on Python.

### Contacts
For questions or support, feel free to reach out:  
**Cuiling Wu** - LDM-GM (graph application):  wucuiling@zhejianglab.com  
**Lu Zhang** - LDM-GM (signed distance field):  luluzhang@zhejianglab.com

## 📁 train database batch
Sample database files containing geometries in two formats (📁 stp,  📁 stl) and mechanical properties: homogenization results in 📊 properties.stl  and stress-strain curve (📁 ssc).

### Contacts
For questions or support, feel free to reach out:  
**Anna Stepashkina** - FEA: anna_step@zhejianglab.com, [RigijAp](https://github.com/RigijAp)

## 📁 mechanical properties 
FEA scripts for Matlab+COMSOL `homogenization.m` for homogenisation properties
**Input:**  
- Material properties: Poisson's ratio, Young's modulus, density.  
- Geometry file in `.stp` or `.step` format.  
**Output:**  
- elasticity tensor; compliance tensor; density; volume  

FEA scripts for Matlab+COMSOL  `stress-strain curve.m` for homogenisation properties during pressing
**Input:**  
- Material properties: Poisson's ratio, Young's modulus, density, plasticity curve (default: TC4 material).  
- Geometry file in `.stp` or `.step` format.  
**Output:**  
- stress values; strain values; z-component of reaction forces.

### Contacts
For questions or support, feel free to reach out:  
**Anna Stepashkina** - FEA: anna_step@zhejianglab.com, [RigijAp](https://github.com/RigijAp)

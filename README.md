# FlowPET: Physics-Informed Symplectic Flow Matching for Low-Count PET Reconstruction

**[ICML 2026]** Official implementation of *FlowPET: Physics-Informed Symplectic Flow Matching for Low-Count PET Reconstruction*.

[[Paper (arXiv)]](#)

> Low-count PET reconstruction is severely hindered by the dissipative nature of prevailing generative models, where the inherent phase-space contraction leads to the numerical extinction ("wash-out") of weak but diagnostically critical lesion signals. To overcome this geometric limitation, we propose **FlowPET**, a physics-informed framework that reformulates reconstruction as volume-preserving transport in a symplectic phase space. By parameterizing the posterior dynamics via a Separable Hamiltonian System, our approach guarantees a divergence-free vector field by construction, theoretically immunizing weak signals against probability mass collapse. We train the model via symplectic flow matching and perform inference using a symplectic leapfrog integrator. Extensive experiments on BrainWeb, clinical pediatric, and UDPET datasets demonstrate that FlowPET not only surpasses state-of-the-art deterministic and stochastic baselines in SSIM and PSNR but, more crucially, exhibits superior recovery of low-contrast lesions.

---

## Highlights

- **Symplectic Generative Framework**: Separable Hamiltonian dynamics ensuring divergence-free transport, theoretically preventing the "signal wash-out" pervasive in dissipative models.
- **Physics-Informed Orthogonality**: Range-Null space decomposition for phase-space boundaries, enforcing data consistency in the range space while confining stochastic exploration to the null space.
- **Structure-Preserving Inference**: Symplectic Leapfrog (Störmer-Verlet) integrator maintaining exact phase-space volume preservation during discrete inference steps.

## Code

🚧 **Code is being prepared and will be released soon.** 🚧

As we are working on its extension, code will be released in about August.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhang2026flowpet,
  title={FlowPET: Physics-Informed Symplectic Flow Matching for Low-Count PET Reconstruction},
  author={Zhang, Zheng and Tang, Hao and Hu, Yingying and Hu, Zhanli and Qin, Jing},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## Related Work

- [FourierPET](https://github.com/xiaochaorouz/FourierPET) — FourierPET: Deep Fourier-based Unrolled Network for Low-count PET Reconstruction **(AAAI 2026 Oral)**.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

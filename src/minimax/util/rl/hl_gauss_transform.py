import chex
import jax
import jax.numpy as jnp
import jax.scipy.special


def hl_gauss_transform(
    min_value: float,
    max_value: float,
    num_bins: int,
    sigma: float,
):
    support = jnp.linspace(min_value, max_value, num_bins+1, dtype=jnp.float32)

    def transform_to_probs(target: chex.Array) -> chex.Array:
        cdf_evals = jax.scipy.special.erf((support-target)/(jnp.sqrt(2)*sigma))
        z = cdf_evals[-1] - cdf_evals[0]
        bin_probs = cdf_evals[1:] - cdf_evals[:-1]
        return bin_probs / z

    def transform_from_probs(probs: chex.Array) -> chex.Array:
        centers = (support[:-1] + support[1:]) / 2
        return jnp.sum(probs * centers)

    return transform_to_probs, transform_from_probs


if __name__ == '__main__':
    transform_to_probs, transform_from_probs = hl_gauss_transform(
        min_value=0,
        max_value=20.0,
        num_bins=10,
        sigma=0.1,
    )

    for r in [0, 3, 20]:
        probs = transform_to_probs(jnp.array(r))
        print(f'Probs for {r}: {probs}')
        print(f'Reconstructed from probs: {transform_from_probs(probs)}')

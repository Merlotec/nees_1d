use utility::indifferent_price;

pub mod allocate;
pub mod utility;

pub trait Agent {
    type FloatType: num::Float;

    fn agent_id(&self) -> usize;
    fn income(&self) -> Self::FloatType;
    fn utility(&self, price: Self::FloatType, quality: Self::FloatType) -> Self::FloatType;
}

pub trait Item {
    type FloatType: num::Float;

    fn quality(&self) -> Self::FloatType;
}

#[derive(Debug, Clone, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Allocation<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>> {
    agent: A,
    item: I,

    price: F,
    utility: F,
}

#[allow(dead_code)]
impl<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>> Allocation<F, A, I> {
    pub fn new(agent: A, item: I, price: F) -> Self {
        let utility = agent.utility(price, item.quality());
        Self {
            agent,
            item,

            price,
            utility,
        }
    }

    pub fn decompose(self) -> (A, I) {
        (self.agent, self.item)
    }

    pub fn agent(&self) -> &A {
        &self.agent
    }

    pub fn agent_mut(&mut self) -> &mut A {
        &mut self.agent
    }

    pub fn item(&self) -> &I {
        &self.item
    }

    pub fn quality(&self) -> F {
        self.item.quality()
    }

    pub fn set_item(&mut self, mut item: I) -> I {
        std::mem::swap(&mut self.item, &mut item);
        self.utility = self.agent.utility(self.price, self.item.quality());
        return item;
    }

    pub fn set_agent(&mut self, mut agent: A) -> A {
        assert!(self.price < agent.income());

        std::mem::swap(&mut self.agent, &mut agent);
        self.utility = self.agent.utility(self.price, self.item.quality());
        return agent;
    }

    pub fn set_agent_and_price(&mut self, mut agent: A, price: F) -> A {
        self.price = price;
        assert!(self.price < agent.income());

        std::mem::swap(&mut self.agent, &mut agent);
        self.utility = self.agent.utility(self.price, self.item.quality());
        return agent;
    }

    fn set_price(&mut self, price: F) {
        assert!(price < self.agent.income());

        self.price = price;
        self.utility = self.agent.utility(self.price, self.item.quality());
    }

    pub fn price(&self) -> F {
        self.price
    }

    pub fn utility(&self) -> F {
        self.utility
    }

    pub fn indifferent_price(&self, quality: F, epsilon: F, max_iter: usize) -> Option<F> {
        let (x_min, x_max) = if quality > self.quality() {
            (self.price, self.agent.income())
        } else {
            (F::zero(), self.price)
        };
        indifferent_price(
            self.agent(),
            quality,
            self.utility,
            x_min,
            x_max,
            epsilon,
            max_iter,
        )
    }

    pub fn prefers(&self, other: &Self, epsilon: F) -> bool {
        self.agent().utility(other.price(), other.quality()) > self.utility() + epsilon
    }

    pub fn is_preferred_by(&self, others: &[Self], epsilon: F) -> bool {
        for other in others {
            if other.prefers(self, epsilon) {
                return true;
            }
        }
        false
    }
}

pub fn verify_solution<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    allocations: &[Allocation<F, A, I>],
    epsilon: F,
    max_iter: usize,
) -> bool {
    let mut valid = true;

    for (i, allocation_i) in allocations.iter().enumerate() {
        let u = allocation_i
            .agent
            .utility(allocation_i.price, allocation_i.item.quality());
        if (u - allocation_i.utility()).abs() > epsilon {
            println!("Agent {} has a utility mismatch!", i);
            return false;
        }

        for (j, allocation_j) in allocations.iter().enumerate() {
            if i != j {
                if allocation_j.agent.agent_id() == allocation_i.agent.agent_id() {
                    println!(
                        "Agent {} has the same item_id as {}; item_id= {}",
                        i,
                        j,
                        allocation_j.agent.agent_id()
                    );
                    valid = false;
                }

                // Compute the utility agent i would get from allocation j
                let u_alt = allocation_i
                    .agent
                    .utility(allocation_j.price, allocation_j.quality());
                if u_alt > u + epsilon {
                    let p_alt = allocation_i
                        .indifferent_price(allocation_j.quality(), epsilon, max_iter)
                        .unwrap();
                    println!(
                        "Agent {} prefers allocation {}, (delta_u = {}, delta_p = {})",
                        i,
                        j,
                        (u_alt - u).to_f32().unwrap(),
                        (p_alt - allocation_j.price()).to_f32().unwrap(),
                    );
                    valid = false;
                }
            }
        }
    }

    valid
}

pub fn favourite<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    agent: &A,
    allocations: &[Allocation<F, A, I>],
    epsilon: F,
) -> Option<(usize, F)> {
    let mut u_max = F::zero();
    let mut fav: Option<usize> = None;
    for (l, other) in allocations.iter().enumerate().rev() {
        if agent.income() > other.price() + epsilon {
            let u = agent.utility(other.price(), other.quality());
            if fav.is_none() || u > u_max + epsilon {
                u_max = u;
                fav = Some(l)
            }
        }
    }

    fav.map(|x| (x, u_max))
}

pub fn min_favourite<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    allocations: &[Allocation<F, A, I>],
    range: std::ops::RangeInclusive<usize>,
    epsilon: F,
) -> Option<usize> {
    let mut fav_min: Option<usize> = None;
    for l in range {
        if let Some((fav, _)) = favourite(allocations[l].agent(), allocations, epsilon) {
            if let Some(min) = &mut fav_min {
                if &fav < min {
                    *min = fav;
                }
            } else {
                fav_min = Some(fav);
            }
        }
    }

    fav_min
}

pub fn max_favourite<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    allocations: &[Allocation<F, A, I>],
    range: std::ops::RangeInclusive<usize>,
    epsilon: F,
) -> Option<usize> {
    let mut fav_max: Option<usize> = None;
    for l in range {
        if let Some((fav, _)) = favourite(allocations[l].agent(), allocations, epsilon) {
            if let Some(min) = &mut fav_max {
                if &fav > min {
                    *min = fav;
                }
            } else {
                fav_max = Some(fav);
            }
        }
    }

    fav_max
}

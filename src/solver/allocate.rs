use super::*;
use crate::solver::{Agent, Allocation, Item};
use std::mem;
/// Represents errors that can occur during the fractal equilibrium computation.
#[derive(Debug, Copy, Clone)]
pub enum AllocationError {
    /// No price found that makes an agent indifferent between two reference qualities.
    NoIndifference,
    /// No suitable agent could be identified for allocation at the required step.
    NoCandidate,
    /// An agent's income was exceeded during allocation. The optional `usize` may identify the agent index.
    IncomeExceeded(Option<usize>),
    /// No boundary could be determined for the current set of allocations.
    NoBoundary,
    /// Attempted to insert an agent into an invalid position (e.g., slot already occupied).
    InvalidInsertion,
    /// Required an intermediate agent for realignment, but none was available.
    NoIntermediateAgent,
    /// An allocation was found empty where an agent was expected.
    EmptyAllocation,
    /// The envelope (a range of allocations subject to reallocation) could not be realigned to a valid state.
    EnvelopeBreach,
}

/// A specialized result type for the fractal module.
pub type AllocationResult<T> = Result<T, AllocationError>;

/// Configuration parameters for fractal computations, such as tolerance and iteration limits.
#[derive(Debug, Clone)]
pub struct AllocationSettings<F: num::Float> {
    /// Convergence tolerance for iterative computations.
    pub epsilon: F,
    /// Maximum number of iterations for convergence attempts.
    pub max_iter: usize,
}

/// A container that can either hold an `Agent` or be empty. This is useful when temporarily
/// displacing agents during the reallocation steps.
#[derive(Debug, Clone)]
pub enum AgentHolder<A: Agent> {
    /// No agent currently held.
    Empty,
    /// An agent is stored here.
    Agent(A),
}

impl<A: Agent> AgentHolder<A> {
    /// Checks if this holder currently contains an agent.
    pub fn has_agent(&self) -> bool {
        matches!(self, AgentHolder::Agent(_))
    }

    /// Removes the agent from this holder and returns it as a new `AgentHolder`.
    pub fn take(&mut self) -> Self {
        let mut other = AgentHolder::Empty;
        mem::swap(self, &mut other);
        other
    }

    /// Converts this `AgentHolder` into an `Option<A>`, consuming it.
    pub fn to_option(self) -> Option<A> {
        if let AgentHolder::Agent(a) = self {
            Some(a)
        } else {
            None
        }
    }

    /// Returns a reference to the contained agent, or panics if empty.
    pub fn agent(&self) -> &A {
        match self {
            AgentHolder::Agent(a) => a,
            AgentHolder::Empty => panic!("Attempted to access an empty AgentHolder."),
        }
    }
}

impl<A: Agent> Agent for AgentHolder<A> {
    type FloatType = A::FloatType;

    fn agent_id(&self) -> usize {
        self.agent().agent_id()
    }

    fn income(&self) -> Self::FloatType {
        self.agent().income()
    }

    fn utility(&self, price: Self::FloatType, quality: Self::FloatType) -> Self::FloatType {
        self.agent().utility(price, quality)
    }
}

/// Direction in which we attempt to resolve allocation conflicts (double-cross scenarios).
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Direction {
    /// Realign by moving allocations toward higher indices.
    Up,
    /// Realign by moving allocations toward lower indices.
    Down,
}

/// Represents an "envelope" of allocations subject to reallocation attempts.
/// The envelope is defined by a range of allocations and a reference (env) point.
#[derive(Debug)]
pub struct Envelope<'a, F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>> {
    /// Slice of allocations forming the envelope.
    pub allocations: &'a mut [Allocation<F, A, I>],
    /// The location of the "envelope agent", often where a double-cross or pivot occurs.
    pub env: usize,
    /// The location of the last agent that forms the starting point of the envelope.
    pub src: usize,
    /// The direction in which we must resolve the envelope (up or down the index range).
    pub dir: Direction,
}

impl<'a, F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>> Envelope<'a, F, A, I> {
    /// Constructs a new Envelope given a slice of allocations, an envelope point, a source point, and a direction.
    pub fn new(
        allocations: &'a mut [Allocation<F, A, I>],
        env: usize,
        src: usize,
        dir: Direction,
    ) -> Self {
        Self {
            allocations,
            env,
            src,
            dir,
        }
    }
}

impl<'a, F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>
    Envelope<'a, F, AgentHolder<A>, I>
{
    /// Attempts to align the envelope so that a valid non-envy allocation is restored.
    ///
    /// This function assumes that removing the endpoint allocation `src` would result in a valid solution,
    /// and tries to reinstate a valid configuration including `src`. If it fails, `FractalError::EnvelopeBreach`
    /// is returned.
    pub fn align(&mut self, settings: &AllocationSettings<F>) -> AllocationResult<()> {
        match self.dir {
            Direction::Up => {
                let start = self.src;
                let mut i = start;
                while i > self.env {
                    match try_align_down(self.allocations, start, i, settings) {
                        Ok(_) => {
                            // Check if the current state remains valid or needs further adjustments.
                            if let Some(fav) =
                                min_favourite(self.allocations, i..=start, settings.epsilon)
                            {
                                if fav <= self.env {
                                    // Allocation preferences breached, cannot maintain a valid envelope.
                                    return Err(AllocationError::EnvelopeBreach);
                                } else if fav < i {
                                    // Need to shift down to favor a lower-index allocation.
                                    i = fav;
                                } else {
                                    // Successfully aligned.
                                    return Ok(());
                                }
                            } else {
                                // No further rearrangements needed.
                                return Ok(());
                            }
                        }
                        Err(AllocationError::IncomeExceeded(_)) => {
                            return Err(AllocationError::EnvelopeBreach)
                        }
                        Err(e) => return Err(e),
                    }
                }
            }
            Direction::Down => {
                let start = self.src;
                let mut i = start;
                while i < self.env {
                    match try_align_up(self.allocations, start, i, settings) {
                        Ok(_) => {
                            // Check if the current state remains valid or needs further adjustments.
                            if let Some(fav) =
                                max_favourite(self.allocations, i..=start, settings.epsilon)
                            {
                                if fav >= self.env {
                                    return Err(AllocationError::EnvelopeBreach);
                                } else if fav > i {
                                    i = fav;
                                } else {
                                    return Ok(());
                                }
                            } else {
                                // No further rearrangements needed.
                                return Ok(());
                            }
                        }
                        Err(AllocationError::IncomeExceeded(_)) => {
                            return Err(AllocationError::EnvelopeBreach)
                        }
                        Err(e) => return Err(e),
                    }
                }
            }
        }
        Err(AllocationError::EnvelopeBreach)
    }
}

/// Moves an agent from one position to another in the allocations array, shifting other agents accordingly.
/// This is used during complex realignments (e.g., promotions or demotions of agents).
pub fn displace<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    allocations: &mut [Allocation<F, AgentHolder<A>, I>],
    i: usize,
    to: usize,
) {
    if i > to {
        // Move agent downwards.
        let mut buffer = allocations[to].agent_mut().take();
        let agent = allocations[i].agent_mut().take();
        allocations[to].set_agent(agent);
        for j in to + 1..=i {
            let hold = allocations[j].agent_mut().take();
            allocations[j].set_agent(buffer.take());
            buffer = hold;
        }
        allocations[i].set_agent(buffer);
    } else if to > i {
        // Move agent upwards.
        let mut buffer = allocations[to].agent_mut().take();
        let agent = allocations[i].agent_mut().take();
        allocations[to].set_agent(agent);
        for j in (i..to).rev() {
            let hold = allocations[j].agent_mut().take();
            allocations[j].set_agent(buffer.take());
            buffer = hold;
        }
    }
}

/// Finds the agent to allocate next when moving "up" (increasing indices) along the quality axis.
/// This is used when q0 < q1 to determine the next agent that achieves a non-envy state.
pub fn next_agent_up<F: num::Float, A: Agent<FloatType = F>>(
    agents: &[A],
    q0: F,
    p0: F,
    q1: F,
    settings: &AllocationSettings<F>,
) -> AllocationResult<(usize, F)> {
    assert!(q1 >= q0);
    let mut p_min = F::zero();
    let mut to_allocate: Option<usize> = None;

    for (a, agent) in agents.iter().enumerate() {
        if agent.income() <= p0 {
            return Err(AllocationError::IncomeExceeded(Some(a)));
        }
        let p_indif = indifferent_price(
            agent,
            q1,
            agent.utility(p0, q0),
            p0,
            agent.income(),
            settings.epsilon,
            settings.max_iter,
        )
        .ok_or(AllocationError::NoIndifference)?;

        if to_allocate.is_none() || p_indif < p_min {
            p_min = p_indif;
            to_allocate = Some(a);
        }
    }

    to_allocate
        .map(|x| (x, p_min))
        .ok_or(AllocationError::NoCandidate)
}

/// Finds the agent to allocate next when moving "down" (decreasing indices) along the quality axis.
/// This is used when q0 > q1 to determine the next agent that achieves a non-envy state.
pub fn next_agent_down<F: num::Float, A: Agent<FloatType = F>>(
    agents: &[A],
    q0: F,
    p0: F,
    q1: F,
    settings: &AllocationSettings<F>,
) -> AllocationResult<(usize, F)> {
    assert!(q1 <= q0);
    let mut p_min = F::zero();
    let mut to_allocate: Option<usize> = None;

    for (a, agent) in agents.iter().enumerate() {
        if agent.income() <= p0 {
            // Normally should not happen. If it does, we cannot use this agent.
            continue;
        }
        let p_indif = indifferent_price(
            agent,
            q1,
            agent.utility(p0, q0),
            F::zero(),
            p0,
            settings.epsilon,
            settings.max_iter,
        )
        .ok_or(AllocationError::NoIndifference)?;

        if to_allocate.is_none() || p_indif < p_min {
            p_min = p_indif;
            to_allocate = Some(a);
        }
    }

    to_allocate
        .map(|x| (x, p_min))
        .ok_or(AllocationError::NoCandidate)
}

/// Attempts to determine a boundary point in the allocations for a given quality.
/// A boundary point corresponds to an allocation where an agent is indifferent at some price.
pub fn partial_boundary<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    allocations: &[Allocation<F, AgentHolder<A>, I>],
    quality: F,
    settings: &AllocationSettings<F>,
) -> Option<(usize, F)> {
    let mut p_max: F = num::zero();
    let mut i_max: Option<usize> = None;

    for (i, alloc) in allocations.iter().enumerate().rev() {
        if alloc.agent().has_agent() {
            if i_max.is_none() {
                p_max = alloc.indifferent_price(quality, settings.epsilon, settings.max_iter)?;
                i_max = Some(i);
            } else {
                // Only consider this allocation if it shows a preference that pushes the boundary further.
                let u_other = alloc.agent().utility(p_max, quality);
                if u_other > alloc.utility() {
                    p_max =
                        alloc.indifferent_price(quality, settings.epsilon, settings.max_iter)?;
                    i_max = Some(i);
                }
            }
        }
    }

    i_max.map(|i| (i, p_max))
}

/// Similar to `partial_boundary` but used in contexts where all allocations are defined.
/// By using this function we are asserting that the boundary is indeed the global boundary of the system.
/// If any agents are `AgentHolder::Empty` then this will panic, so it provides an implicit assertion
/// that all allocations actually contain agents.
pub fn boundary<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    allocations: &[Allocation<F, A, I>],
    quality: F,
    settings: &AllocationSettings<F>,
) -> Option<(usize, F)> {
    let mut p_max: F = num::zero();
    let mut i_max: Option<usize> = None;

    for (i, alloc) in allocations.iter().enumerate().rev() {
        if i_max.is_none() {
            p_max = alloc.indifferent_price(quality, settings.epsilon, settings.max_iter)?;
            i_max = Some(i);
        } else {
            let u_other = alloc.agent().utility(p_max, quality);
            if u_other > alloc.utility() {
                p_max = alloc.indifferent_price(quality, settings.epsilon, settings.max_iter)?;
                i_max = Some(i);
            }
        }
    }

    i_max.map(|i| (i, p_max))
}

/// Restores a set of agents back into allocations after a failed attempt at realignment.
/// This is used as a rollback mechanism.
pub fn recover_agents<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    allocations: &mut [Allocation<F, AgentHolder<A>, I>],
    agents: Vec<A>,
    dir: Direction,
) {
    assert_eq!(allocations.len(), agents.len());
    match dir {
        Direction::Down => {
            // Place agents starting from the end going backward.
            for (i, agent) in agents.into_iter().enumerate() {
                allocations[allocations.len() - i - 1].set_agent(AgentHolder::Agent(agent));
            }
        }
        Direction::Up => {
            // Place agents starting from the beginning going forward.
            for (i, agent) in agents.into_iter().enumerate() {
                allocations[i].set_agent(AgentHolder::Agent(agent));
            }
        }
    }
}

/// Attempts to align allocations in a downward direction between `start` and `end`.
/// This involves extracting agents, selecting the next agent to allocate, and potentially
/// performing recursive calls to ensure non-envy conditions are met.
pub fn try_align_down<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    allocations: &mut [Allocation<F, AgentHolder<A>, I>],
    start: usize,
    end: usize,
    settings: &AllocationSettings<F>,
) -> AllocationResult<usize> {
    assert!(end <= start);

    // Extract agents currently between `end` and `start`.
    let agent_holders: Vec<AgentHolder<A>> = allocations[end..=start]
        .iter_mut()
        .map(|alloc| alloc.agent.take())
        .collect();

    let mut agents = Vec::with_capacity(agent_holders.len());
    for ah in agent_holders {
        agents.push(ah.to_option().ok_or(AllocationError::NoIntermediateAgent)?);
    }

    let mut last_b = None;
    for l in (end..=start).rev() {
        let q0 = allocations[l].quality();
        let (inner_b, p0) =
            partial_boundary(allocations, q0, settings).ok_or(AllocationError::NoBoundary)?;

        let agent = if l > end {
            // Select the next agent to move down.
            let q1 = allocations[l - 1].quality();
            let (a, _) = match next_agent_down(&agents, q0, p0, q1, settings) {
                Ok(x) => x,
                Err(AllocationError::NoCandidate) | Err(AllocationError::IncomeExceeded(_)) => {
                    recover_agents(&mut allocations[end..=l], agents, Direction::Down);
                    return Err(AllocationError::IncomeExceeded(None));
                }
                Err(e) => return Err(e),
            };
            agents.remove(a)
        } else {
            // Last agent to place: no further checks needed.
            agents.pop().unwrap()
        };

        allocations[l].set_agent_and_price(AgentHolder::Agent(agent), p0);
        allocate(allocations, l, Some(inner_b), Direction::Down, settings)?;
        last_b = Some(inner_b);
    }

    Ok(last_b.unwrap())
}

/// Attempts to align allocations in an upward direction between `start` and `end`.
/// Similar in function to `try_align_down` but acts in the opposite direction.
fn try_align_up<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    allocations: &mut [Allocation<F, AgentHolder<A>, I>],
    start: usize,
    end: usize,
    settings: &AllocationSettings<F>,
) -> AllocationResult<usize> {
    assert!(end >= start);

    let agent_holders: Vec<AgentHolder<A>> = allocations[start..=end]
        .iter_mut()
        .map(|alloc| alloc.agent.take())
        .collect();

    let mut agents = Vec::with_capacity(agent_holders.len());
    for ah in agent_holders {
        agents.push(ah.to_option().ok_or(AllocationError::NoIntermediateAgent)?);
    }

    let mut last_b = None;
    for l in start..=end {
        let q0 = allocations[l].quality();
        let (inner_b, p0) =
            partial_boundary(allocations, q0, settings).ok_or(AllocationError::NoBoundary)?;

        let agent = if l < end {
            // Identify which agent to allocate next going upwards.
            let q1 = allocations[l + 1].quality();
            let (a, _) = match next_agent_up(&agents, q0, p0, q1, settings) {
                Ok(x) => x,
                Err(AllocationError::NoCandidate) => {
                    recover_agents(&mut allocations[l..=end], agents, Direction::Up);
                    return Err(AllocationError::IncomeExceeded(None));
                }
                Err(e) => return Err(e),
            };
            agents.remove(a)
        } else {
            // Last agent to allocate upward.
            agents.pop().unwrap()
        };

        allocations[l].set_agent_and_price(AgentHolder::Agent(agent), p0);
        allocate(allocations, l, Some(inner_b), Direction::Up, settings)?;
        last_b = Some(inner_b);
    }

    Ok(last_b.unwrap())
}

/// The recursive allocation function that ensures no-envy conditions are maintained.
/// It attempts to place an agent at allocation index `i`, potentially resolving double-crosses
/// by rearranging envelopes above or below, depending on the direction.
pub fn allocate<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    allocations: &mut [Allocation<F, AgentHolder<A>, I>],
    i: usize,
    boundary: Option<usize>,
    dir: Direction,
    settings: &AllocationSettings<F>,
) -> AllocationResult<()> {
    if let Some(b) = boundary {
        if i < allocations.len() {
            if let AgentHolder::Empty = allocations[i].agent() {
                return Err(AllocationError::InvalidInsertion);
            }
        } else {
            return Err(AllocationError::InvalidInsertion);
        }

        if i.abs_diff(b) > 0 {
            // We have detected a double-cross scenario between i and b.
            // Need to remove allocations in the range and attempt a controlled reallocation.
            if b < i && dir == Direction::Up {
                // Attempt resolving by promoting the agent at b.
                let promote = {
                    let mut envelope = Envelope::new(allocations, b, i, Direction::Up);
                    match envelope.align(settings) {
                        Ok(_) => false,
                        Err(AllocationError::EnvelopeBreach) => true,
                        Err(e) => return Err(e),
                    }
                };

                if promote {
                    // Promote the agent at b if envelope alignment failed.
                    displace(allocations, b, i);
                    let to_promote = allocations[i].agent_mut().take();
                    try_align_up(allocations, b, i - 1, settings)?;
                    let (b_promoted, p_promoted) =
                        partial_boundary(allocations, allocations[i].quality(), settings)
                            .ok_or(AllocationError::NoBoundary)?;
                    allocations[i].set_agent_and_price(to_promote, p_promoted);
                    allocate(allocations, i, Some(b_promoted), Direction::Up, settings)?;
                }
            } else if i < b && dir == Direction::Down {
                // Attempt resolving by demoting the agent at b.
                let demote = {
                    let mut envelope = Envelope::new(allocations, b, i, Direction::Down);
                    match envelope.align(settings) {
                        Ok(_) => false,
                        Err(AllocationError::EnvelopeBreach) => true,
                        Err(e) => return Err(e),
                    }
                };

                if demote {
                    // Demote the agent at b if envelope alignment failed.
                    displace(allocations, b, i);
                    let to_demote = allocations[i].agent_mut().take();
                    try_align_down(allocations, b, i + 1, settings)?;
                    let (b_demoted, p_demoted) =
                        partial_boundary(allocations, allocations[i].quality(), settings)
                            .ok_or(AllocationError::NoBoundary)?;
                    allocations[i].set_agent_and_price(to_demote, p_demoted);
                    allocate(allocations, i, Some(b_demoted), Direction::Down, settings)?;
                }
            }
        }
    }

    Ok(())
}

/// The entry point for computing a non-envy equilibrium allocation of agents to items.
/// - `agents`: a vector of agents (with heterogeneous utilities and incomes).
/// - `items`: a vector of items (with heterogeneous qualities).
/// - `settings`: fractal computation settings (tolerance, iterations).
/// - `render_pipe`: optional pipe for rendering allocations as they evolve.
///
/// Returns a vector of `Allocation` representing a final non-envy equilibrium if successful.
pub fn allocate_all<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(
    mut agents: Vec<A>,
    mut items: Vec<I>,
    constraint_price: F,
    settings: AllocationSettings<F>,
) -> AllocationResult<Vec<Allocation<F, A, I>>> {
    assert_eq!(agents.len(), items.len());

    let mut allocations: Vec<Allocation<F, AgentHolder<A>, I>> = Vec::new();

    // Main allocation loop: each iteration picks an item and finds the appropriate agent to allocate.
    while !items.is_empty() {
        let q0 = items.first().unwrap().quality();
        let (b, p0) = if allocations.is_empty() {
            // On the first allocation, start with a zero price and no boundary.
            (None, constraint_price)
        } else {
            // Find the boundary for the next quality.
            let (b, p1) =
                boundary(&allocations, q0, &settings).ok_or(AllocationError::NoBoundary)?;
            (Some(b), p1)
        };

        let a = if items.len() > 1 {
            // Find the agent that ensures a non-envy equilibrium with the next item's quality.
            match next_agent_up(&agents, q0, p0, items[1].quality(), &settings) {
                Ok((a, _)) => a,
                Err(AllocationError::IncomeExceeded(Some(a))) => a,
                Err(e) => return Err(e),
            }
        } else {
            // Only one agent and one item left, must be allocated directly.
            0
        };

        let new_allocation =
            Allocation::new(AgentHolder::Agent(agents.remove(a)), items.remove(0), p0);
        let i = allocations.len();
        allocations.push(new_allocation);

        // Recursively ensure no-envy conditions by aligning envelopes if needed.
        allocate(&mut allocations, i, b, Direction::Up, &settings)?;
    }

    // Final clean-up: extract the actual agents and items from `AgentHolder` and return the final solution.
    let mut cleaned = Vec::with_capacity(allocations.len());
    for alloc in allocations {
        let p = alloc.price();
        let (mut agent, item) = alloc.decompose();
        cleaned.push(Allocation::new(
            agent
                .take()
                .to_option()
                .ok_or(AllocationError::EmptyAllocation)?,
            item,
            p,
        ));
    }

    Ok(cleaned)
}

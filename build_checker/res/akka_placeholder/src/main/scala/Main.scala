// UltraComplexFSMClusterExample.scala

/*
  This scenario is designed to show an extremely sophisticated combination:

  1. A typed FSM that has multiple states: Idle, Working, WaitingForCluster.
  2. It interacts with a cluster, waiting for a minimum number of members.
  3. Once the cluster is large enough, it transitions to a Working state.
  4. In the Working state, it can accept tasks, possibly store them in memory.
  5. Then it can move to an Idle state if a pause message is received.
  6. We demonstrate advanced usage of timers, cluster membership checks, and typed FSM illusions via Behaviors.

  Because typed FSM is not built-in, we simulate an FSM by manually handling states and transitions with become.
*/

import akka.actor.typed.scaladsl.{Behaviors, ActorContext, TimerScheduler}
import akka.actor.typed.{ActorRef, ActorSystem, Behavior}
import akka.cluster.typed.{Cluster, Subscribe, SelfUp, MemberUp, MembershipEvent}
import scala.concurrent.duration._

object UltraComplexFSM {
  // Sealed trait for commands
  sealed trait Command

  // Commands for cluster-based transitions
  case object CheckClusterSize extends Command
  case object StartWork extends Command
  case object PauseWork extends Command
  case class AddTask(task: String) extends Command
  case class PrintTasks(replyTo: ActorRef[String]) extends Command
  private case class MemberUpEvent(evt: MemberUp) extends Command

  // We'll define a config param for min cluster size
  // so we only start working if cluster members >= that.
  final case class Config(minClusterSize: Int)

  def apply(config: Config): Behavior[Command] = Behaviors.setup { ctx =>
    // We'll subscribe to cluster membership events
    val cluster = Cluster(ctx.system)
    cluster.subscriptions ! Subscribe(ctx.self, classOf[MemberUp])

    // We start in the "WaitingForCluster" state
    waitingForCluster(config, membersCount = cluster.state.members.size, tasks = List.empty)
  }

  // waitingForCluster: we do not accept tasks or start real work until we have enough members
  private def waitingForCluster(
    config: Config,
    membersCount: Int,
    tasks: List[String]
  ): Behavior[Command] = Behaviors.receive { (ctx, msg) =>
    msg match {
      case MemberUpEvent(MemberUp(_)) =>
        val newCount = membersCount + 1
        ctx.log.info(s"A new member joined! Cluster size = $newCount")
        if (newCount >= config.minClusterSize) {
          ctx.log.info("Minimum cluster size satisfied. Transitioning to Idle.")
          idle(config, newCount, tasks)
        } else {
          waitingForCluster(config, newCount, tasks)
        }

      case AddTask(task) =>
        // We can store tasks even though not operational, or ignore them.
        val updated = tasks :+ task
        ctx.log.info(s"Storing task while waiting for cluster: $task")
        waitingForCluster(config, membersCount, updated)

      case PrintTasks(replyTo) =>
        replyTo ! s"Tasks while waiting: ${tasks.mkString(", ")}, cluster size = $membersCount"
        Behaviors.same

      case _ => Behaviors.same
    }
  }

  // Once we have enough cluster members, we move to an Idle state.
  private def idle(
    config: Config,
    membersCount: Int,
    tasks: List[String]
  ): Behavior[Command] = Behaviors.receive { (ctx, msg) =>
    msg match {
      case StartWork =>
        ctx.log.info("Transitioning from Idle to Working.")
        working(config, membersCount, tasks)

      case AddTask(task) =>
        // We can accumulate tasks while idle.
        val updated = tasks :+ task
        ctx.log.info(s"Task queued while idle: $task")
        idle(config, membersCount, updated)

      case PrintTasks(replyTo) =>
        replyTo ! s"Idle tasks: ${tasks.mkString(", ")}, cluster size = $membersCount"
        Behaviors.same

      case MemberUpEvent(MemberUp(_)) =>
        // cluster grew, update count
        val newCount = membersCount + 1
        ctx.log.info(s"New member joined while idle, cluster size = $newCount")
        idle(config, newCount, tasks)

      case _ => Behaviors.same
    }
  }

  // In working state, we can accept tasks and process them, or pause to return to idle.
  private def working(
    config: Config,
    membersCount: Int,
    tasks: List[String]
  ): Behavior[Command] = Behaviors.receive { (ctx, msg) =>
    msg match {
      case AddTask(task) =>
        // Let's consider the tasks are immediately processed.
        ctx.log.info(s"Working on task: $task")
        working(config, membersCount, tasks :+ task)

      case PrintTasks(replyTo) =>
        replyTo ! s"Working tasks so far: ${tasks.mkString(", ")}, cluster size = $membersCount"
        Behaviors.same

      case PauseWork =>
        ctx.log.info("Pausing work, transitioning to Idle.")
        idle(config, membersCount, tasks)

      case MemberUpEvent(MemberUp(_)) =>
        // cluster grew in the meantime
        val newCount = membersCount + 1
        ctx.log.info(s"Cluster gained a new member, total = $newCount while working.")
        working(config, newCount, tasks)

      case _ => Behaviors.same
    }
  }
}

object UltraComplexFSMClusterApp extends App {
  import UltraComplexFSM._

  // We'll define a config requiring a cluster of size 2 before we start.
  val system = ActorSystem(UltraComplexFSM(Config(minClusterSize = 2)), "UltraFSMClusterSystem")

  // For local testing without an actual cluster, we can forcibly inject a MemberUpEvent for demonstration:
  system ! MemberUpEvent(null) // Pretend we have 1 member

  // Try adding tasks, see them stored while not enough members.
  system ! AddTask("Task1")
  system ! AddTask("Task2")

  // Force a second member join event
  system ! MemberUpEvent(null)

  // Now we should transition to Idle, let's add tasks in Idle
  system ! AddTask("IdleTask1")

  // Move to working
  system ! StartWork
  system ! AddTask("WorkTask1")

  // Pause
  Thread.sleep(1000)
  system ! PauseWork

  Thread.sleep(2000)
  system.terminate()
}

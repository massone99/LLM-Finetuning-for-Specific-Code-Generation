// TypedStashExample.scala

import akka.actor.typed.scaladsl.{Behaviors, StashBuffer}
import akka.actor.typed.{ActorSystem, Behavior}

object StashActor {
  sealed trait Command
  case object Ready extends Command
  case class Payload(msg: String) extends Command

  def apply(): Behavior[Command] = Behaviors.withStash(10) { buffer =>
    Behaviors.setup { context =>

      def unstashedBehavior: Behavior[Command] = Behaviors.receiveMessage {
        case Payload(m) =>
          context.log.info(s"Processing payload: $m")
          Behaviors.same
        case Ready =>
          context.log.info("Already in ready state.")
          Behaviors.same
      }

      // Initially not ready
      val initial: Behavior[Command] = Behaviors.receiveMessage {
        case Ready =>
          context.log.info("Becoming ready, unstashing messages.")
          buffer.unstashAll(unstashedBehavior)
        case other =>
          context.log.info("Stashing message until we get Ready.")
          buffer.stash(other)
          Behaviors.same
      }

      initial
    }
  }
}

object TypedStashExampleApp extends App {
  val system = ActorSystem(StashActor(), "StashSystem")

  system ! StashActor.Payload("msg1")
  system ! StashActor.Payload("msg2")
  system ! StashActor.Ready
  system ! StashActor.Payload("msg3")

  Thread.sleep(500)
  system.terminate()
}
